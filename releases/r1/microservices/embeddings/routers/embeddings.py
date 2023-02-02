import tensorflow_hub as hub
from fastapi import APIRouter, Body

from schemas.emebeddings import EmbeddingsRequest, EmbeddingsResponse

router = APIRouter(prefix="/embeddings")

# USEv5 is about 100x faster than 4
EMBED = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


@router.post("/", response_model=EmbeddingsResponse)
async def embed(
    payload: EmbeddingsRequest = Body(
        ..., example={"strings": ["test 1", "test 2", "foo", "bar"]}
    )
) -> EmbeddingsResponse:
    """Embeds a list of strings, returning a list of floats for every string input"""
    embeddings = [i.numpy().tolist() for i in EMBED(payload.strings)]
    return EmbeddingsResponse(embeddings=embeddings, **payload.dict())
