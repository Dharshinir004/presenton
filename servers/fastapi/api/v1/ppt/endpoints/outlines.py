import asyncio
import json
import math
import traceback
import uuid
import dirtyjson
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from models.presentation_outline_model import PresentationOutlineModel
from models.sql.presentation import PresentationModel
from models.sse_response import (
    SSECompleteResponse,
    SSEErrorResponse,
    SSEResponse,
    SSEStatusResponse,
)
from services.temp_file_service import TEMP_FILE_SERVICE
from services.database import get_async_session
from services.documents_loader import DocumentsLoader
from utils.llm_calls.generate_presentation_outlines import generate_ppt_outline
from utils.ppt_utils import get_presentation_title_from_outlines

OUTLINES_ROUTER = APIRouter(prefix="/outlines", tags=["Outlines"])


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and validate JSON from potentially malformed model response
    """
    if not response_text or response_text.strip() == "":
        raise ValueError("Empty response from model")
    
    cleaned = response_text.strip()
    
    # Remove markdown code blocks if present
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    
    # Remove any text before the first { and after the last }
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}') + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in response")
    
    json_str = cleaned[start_idx:end_idx].strip()
    
    # Try strict JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed: {e}")
        # Fallback to dirtyjson with better error handling
        try:
            return dirtyjson.loads(json_str)
        except Exception as dirty_error:
            print(f"DirtyJSON parsing also failed: {dirty_error}")
            print(f"Problematic JSON string: {json_str[:500]}...")
            raise ValueError(f"Failed to parse JSON response: {dirty_error}")


def validate_presentation_outlines(data: dict) -> bool:
    """
    Validate that the parsed data has the expected structure
    """
    if not isinstance(data, dict):
        return False
    
    if "slides" not in data:
        return False
    
    if not isinstance(data["slides"], list):
        return False
    
    # Basic validation of slide structure
    for slide in data["slides"]:
        if not isinstance(slide, dict):
            return False
        if "title" not in slide or "content" not in slide:
            return False
        if not isinstance(slide["title"], str):
            return False
        if not isinstance(slide["content"], list):
            return False
    
    return True


@OUTLINES_ROUTER.get("/stream/{id}")
async def stream_outlines(
    id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    presentation = await sql_session.get(PresentationModel, id)

    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()

    async def inner():
        yield SSEStatusResponse(
            status="Generating presentation outlines..."
        ).to_string()

        additional_context = ""
        if presentation.file_paths:
            documents_loader = DocumentsLoader(file_paths=presentation.file_paths)
            await documents_loader.load_documents(temp_dir)
            documents = documents_loader.documents
            if documents:
                additional_context = "\n\n".join(documents)

        presentation_outlines_text = ""

        n_slides_to_generate = presentation.n_slides
        if presentation.include_table_of_contents:
            needed_toc_count = math.ceil((presentation.n_slides - 1) / 10)
            n_slides_to_generate -= math.ceil(
                (presentation.n_slides - needed_toc_count) / 10
            )

        # Retry logic for outline generation
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                presentation_outlines_text = ""
                async for chunk in generate_ppt_outline(
                    presentation.content,
                    n_slides_to_generate,
                    presentation.language,
                    additional_context,
                    presentation.tone,
                    presentation.verbosity,
                    presentation.instructions,
                    presentation.include_title_slide,
                    presentation.web_search,
                ):
                    if isinstance(chunk, HTTPException):
                        raise chunk
                    
                    presentation_outlines_text += chunk
                    # Stream the raw chunks to the client for visual feedback
                    yield SSEResponse(
                        event="response",
                        data=json.dumps({"type": "chunk", "chunk": chunk}),
                    ).to_string()
                
                # Check if we got valid content
                if presentation_outlines_text.strip():
                    print(f"Attempt {attempt + 1} received content, length: {len(presentation_outlines_text)}")
                    
                    # Try to parse and validate the JSON
                    try:
                        presentation_outlines_json = extract_json_from_response(presentation_outlines_text)
                        
                        # Validate the structure
                        if validate_presentation_outlines(presentation_outlines_json):
                            print(f"Successfully parsed and validated JSON on attempt {attempt + 1}")
                            break  # Success - exit retry loop
                        else:
                            print(f"Invalid structure on attempt {attempt + 1}")
                            presentation_outlines_text = ""  # Reset for retry
                    
                    except Exception as parse_error:
                        print(f"JSON parsing failed on attempt {attempt + 1}: {parse_error}")
                        presentation_outlines_text = ""  # Reset for retry
                
                # If we're on the last attempt and still have issues, don't retry
                if attempt == max_attempts - 1 and not presentation_outlines_text.strip():
                    yield SSEErrorResponse(
                        detail="Failed to generate valid presentation outlines after multiple attempts. The model may be returning invalid JSON format."
                    ).to_string()
                    return
                
                # Wait before retry (except on last attempt)
                if attempt < max_attempts - 1 and not presentation_outlines_text.strip():
                    print(f"Retrying outline generation... Attempt {attempt + 2}/{max_attempts}")
                    yield SSEStatusResponse(
                        status=f"Retrying outline generation... (Attempt {attempt + 2}/{max_attempts})"
                    ).to_string()
                    await asyncio.sleep(2)  # Wait 2 seconds before retry

            except HTTPException as e:
                yield SSEErrorResponse(detail=e.detail).to_string()
                return
            except Exception as e:
                print(f"Unexpected error during outline generation attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:  # Last attempt
                    yield SSEErrorResponse(
                        detail=f"Unexpected error during outline generation: {str(e)}"
                    ).to_string()
                    return
                await asyncio.sleep(2)  # Wait before retry

        # If we exhausted all attempts without success
        if not presentation_outlines_text.strip():
            yield SSEErrorResponse(
                detail="Failed to generate presentation outlines after multiple attempts. Please try again later."
            ).to_string()
            return

        # Final parsing (should succeed if we reached here)
        presentation_outlines_json = None
        try:
            presentation_outlines_json = extract_json_from_response(presentation_outlines_text)
            
            # Final validation
            if not validate_presentation_outlines(presentation_outlines_json):
                yield SSEErrorResponse(
                    detail="Generated outline has invalid structure. Please try again."
                ).to_string()
                return
                
        except Exception as e:
            traceback.print_exc()
            yield SSEErrorResponse(
                detail=f"Failed to parse presentation outlines: {str(e)}",
            ).to_string()
            return
        
        # Create the model instance
        try:
            presentation_outlines = PresentationOutlineModel(**presentation_outlines_json)
        except Exception as e:
            traceback.print_exc()
            yield SSEErrorResponse(
                detail=f"Failed to create presentation outline model: {str(e)}"
            ).to_string()
            return

        # Limit slides to requested number
        presentation_outlines.slides = presentation_outlines.slides[:n_slides_to_generate]

        # Save outlines as soon as they are generated and parsed
        try:
            presentation.outlines = presentation_outlines.model_dump(mode="json")
            presentation.title = get_presentation_title_from_outlines(presentation_outlines)

            sql_session.add(presentation)
            await sql_session.commit()
            
            yield SSECompleteResponse(
                key="presentation", value=presentation.model_dump(mode="json")
            ).to_string()
            
        except Exception as e:
            traceback.print_exc()
            yield SSEErrorResponse(
                detail=f"Failed to save presentation: {str(e)}"
            ).to_string()
            return

    return StreamingResponse(inner(), media_type="text/event-stream")