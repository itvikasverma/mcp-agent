import os
import hashlib
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointIdsList, Filter, FieldCondition, MatchValue,
    VectorParams, Distance, OptimizersConfigDiff
)
from core.config import Config
import json
 
# ───────────── Load Environment ───────────── #
load_dotenv()
 
app = FastAPI()
config = Config()  
 
# ───────────── Setup Logging ───────────── #
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
 
# ───────────── Initialize Qdrant & Embeddings ───────────── #
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL")+":6333")
 
embedding_model = HuggingFaceEmbeddings(model_name=config.MODEL_NAME)
 
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=config.COLLECTION_NAME,
    embeddings=embedding_model
)
 
 
# ───────────── Pydantic Models ───────────── #
class DocumentIn(BaseModel):
    content: str
    doc_title: str
    url: str
    filename: str
    source: str
    creation_date: Optional[str] = None
    file_hash: Optional[str] = None # ⬅️ NEW
 
class SearchIn(BaseModel):
    text: str
    source: Optional[str] = None
    doc_title: Optional[str] = None
    k: Optional[int] = 5
    threshold: Optional[float] = 0.1
 
 
class CountIn(BaseModel):
    source: Optional[str] = None
   
class ListIn(BaseModel):
    source: Optional[str] = None
   
 
@app.get("/")
def ping():
    return {"status": "ok"}
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
def validate_document(doc: DocumentIn):
    required_fields = ["content", "doc_title", "url", "filename", "source"]
    missing = [field for field in required_fields if not getattr(doc, field, None)]
    if missing:
        return False, f"Missing or empty required fields: {', '.join(missing)}"
    return True, None
 
# ───────────── Routes ───────────── #
 
@app.get("/init_qdrant")
def init_qdrant():
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == config.COLLECTION_NAME for c in collections)

        if exists:
            qdrant_client.delete_collection(config.COLLECTION_NAME)

        qdrant_client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=5000,
                memmap_threshold=20000,
                default_segment_number=2
            )
        )

        return {"message": "Qdrant collection recreated successfully"}, 200

    except Exception as e:
        logging.exception("Qdrant Init failed")
        return {"error": str(e)}, 500
 
@app.get("/get_hash")
def get_hash(source: Optional[str] = Query(None)):
    try:
        if not source:
            raise HTTPException(status_code=400, detail="Source missing.")
 
        # --- Qdrant Scrolling Setup ---
        scroll_filter = Filter(
            must=[FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source.strip())
            )]
        )
 
        all_points = []
        next_offset = None
       
        # Qdrant recommended way to retrieve all points: loop with 'offset'
        while True:
            # Use a reasonable limit for each batch (e.g., 1000)
            scroll_res, next_offset = qdrant_client.scroll(
                collection_name=config.COLLECTION_NAME,
                with_payload=True,
                scroll_filter=scroll_filter,
                limit=1000,  # Fetch points in batches of 1000
                offset=next_offset  # Start from the previous offset
            )
           
            all_points.extend(scroll_res)
           
            # Stop if the offset is None (end of collection reached)
            if next_offset is None:
                break
       
        # --- Process Retrieved Points ---
        files_by_name = {}
 
        for d in all_points:
            meta = d.payload.get("metadata", {})
            filename = meta.get("filename", "unknown")
 
            # last one wins; or add check if filename=="unknown"
            files_by_name[filename] = {
                "file_hash": meta.get("file_hash", None),
            }
 
        # --- Persist Snapshot ---
        data = {
            "source": source,
            "file_count": len(files_by_name),
            "files": files_by_name,
        }
        path = f"{source}.json"
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
 
        # --- Prepare Response ---
        files_list = [
            {"filename": fname, **info}
            for fname, info in files_by_name.items()
        ]
 
        return {
            "source": source,
            "file_count": len(files_list),
            "files": files_list,
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/add_doc")
def add_document(doc_in: DocumentIn):
    """
    Ingestion endpoint:
 
    New semantics:
      - {source}.json is a kill-list built by /get_hash.
      - Any file we *want to keep* in VDB must be REMOVED from that JSON here.
      - /sync_deleted will later delete whatever remains in the JSON.
 
    Still keeps:
      - unchanged-file skip
      - content-changed / rename: delete old chunks + reindex
    """
    logging.info(f"[ADD_DOC] Incoming → source={doc_in.source}, filename={doc_in.filename}")
 
    if not doc_in.file_hash:
        raise HTTPException(status_code=400, detail="file_hash is required")
 
    new_hash = doc_in.file_hash
    filename = doc_in.filename
    source = doc_in.source
    json_path = f"{source}.json"
 
    # 1) Load snapshot (kill-list produced by /get_hash)
    snapshot = {"source": source, "files": {}}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                snapshot = json.load(f)
        except Exception as e:
            logging.warning(f"[ADD_DOC] Failed to read snapshot JSON ({json_path}): {e}")
 
    files = snapshot.get("files", {})
    existing_hashes = {info["file_hash"] for info in files.values() if "file_hash" in info}
    filename_exists = filename in files
    old_hash_for_filename = files.get(filename, {}).get("file_hash")
 
    # ────────────────────────────────
    # 2) Decision Logic
    # ────────────────────────────────
 
    # CASE B — Same filename, content unchanged 🟢
    # We want to keep this file → remove it from kill-list, skip reindex.
    if filename_exists and old_hash_for_filename == new_hash:
        logging.info("[ADD_DOC] Skipped (unchanged file, same hash) → removing from kill-list")
        if filename in files:
            del files[filename]
            snapshot["files"] = files

            # 🔒 ATOMIC WRITE
            tmp_path = f"{json_path}.tmp"
            with open(tmp_path, "w") as tmp:
                json.dump(snapshot, tmp, indent=4)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, json_path)
 
        return {
            "message": "Skipped (unchanged)",
            "skipped": True,
            "filename": filename,
            "file_hash": new_hash,
        }
 
    # CASES A & C — Renamed or Content Changed 🗑️ delete old chunks
    is_rename = (not filename_exists) and (new_hash in existing_hashes)
    is_content_change = filename_exists and (old_hash_for_filename != new_hash)
 
    files_to_delete = []
 
    if is_rename:
        logging.info("[ADD_DOC] Detected rename → deleting old name's chunks")
        # Old filenames that had this hash
        files_to_delete = [fn for fn, v in files.items() if v.get("file_hash") == new_hash]
 
    if is_content_change:
        logging.info("[ADD_DOC] Content updated → deleting old chunks for same filename")
        files_to_delete.append(filename)
 
    # Delete old chunks for all identified filenames
    if files_to_delete:
        for fn_to_delete in set(files_to_delete):  # avoid duplicates
            file_filter = Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=source)),
                    FieldCondition(key="metadata.filename", match=MatchValue(value=fn_to_delete)),
                ]
            )
 
            ids = []
            next_offset = None
            while True:
                points, next_offset = qdrant_client.scroll(
                    collection_name=config.COLLECTION_NAME,
                    scroll_filter=file_filter,
                    with_payload=False,
                    limit=1000,
                    offset=next_offset,
                )
                ids.extend([p.id for p in points])
                if next_offset is None:
                    break
 
            if ids:
                qdrant_client.delete(
                    collection_name=config.COLLECTION_NAME,
                    points_selector=PointIdsList(points=ids),
                )
                logging.info(f"[ADD_DOC] Deleted {len(ids)} old chunks for {fn_to_delete}")
 
            # 🔴 IMPORTANT: old filename should NOT be deleted by /sync_deleted now,
            # because we already handled it. Remove from kill-list.
            if fn_to_delete in files:
                del files[fn_to_delete]
 
    # CASE D — New file or fallthrough from A/C
    logging.info("[ADD_DOC] Indexing new or updated file (this file must be kept in VDB).")
 
    # ────────────────────────────────
    # 3) Index chunks (batched)
    # ────────────────────────────────
    now = datetime.utcnow().isoformat()
    doc = Document(
        page_content=doc_in.content,
        metadata={
            "doc_title": doc_in.doc_title,
            "url": doc_in.url,
            "filename": filename,
            "source": source,
            "creation_date": now,
            "added_at": now,
            "last_modified": now,
            "file_hash": new_hash,
        },
    )
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    chunks = splitter.split_documents([doc])
    indexed_chunks = 0
    batch_size = 50
 
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids = [hashlib.md5(c.page_content.encode("utf-8")).hexdigest() for c in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        indexed_chunks += len(batch)
 
    # ────────────────────────────────
    # 4) Mark this file as “kept” in kill-list
    # ────────────────────────────────
    # Any file we want to keep in VDB must NOT remain in the JSON.
    if filename in files:
        del files[filename]
 
    snapshot["files"] = files

    # 🔒 ATOMIC WRITE
    tmp_path = f"{json_path}.tmp"
    with open(tmp_path, "w") as tmp:
        json.dump(snapshot, tmp, indent=4)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, json_path)
 
    logging.info(f"[ADD_DOC] Snapshot updated (kept file removed from kill-list) → {json_path}")
 
    return {
        "message": "Indexed",
        "skipped": False,
        "chunks_stored": indexed_chunks,
        "filename": filename,
        "file_hash": new_hash,
    }

 
 
@app.get("/sync_deleted")
def sync_deleted(source: Optional[str] = Query(None)):
    """
    Delete files from Qdrant that are still present in the snapshot JSON.
 
    New semantics:
      - {source}.json is a kill-list built by /get_hash.
      - /add_doc removes any file that we want to KEEP from this JSON.
      - Whatever is still in JSON when /sync_deleted runs will be DELETED.
    """
 
    try:
        if not source:
            raise HTTPException(status_code=400, detail="Source missing.")
 
        source = source.strip()
        path = f"{source}.json"
 
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Snapshot file '{path}' not found.")
 
        with open(path, "r") as f:
            data = json.load(f)
 
        files_map = data.get("files", {})
        if not isinstance(files_map, dict):
            raise HTTPException(status_code=400, detail="'files' in snapshot must be an object/dict.")
 
        filenames = list(files_map.keys())
 
        total_deleted = 0
        deleted_details = []
 
        for fname in filenames:
            # For each filename still on the kill-list → delete all its chunks
            file_filter = Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=source)),
                    FieldCondition(key="metadata.filename", match=MatchValue(value=fname)),
                ]
            )
 
            ids = []
            next_offset = None
            while True:
                points, next_offset = qdrant_client.scroll(
                    collection_name=config.COLLECTION_NAME,
                    scroll_filter=file_filter,
                    with_payload=False,
                    limit=1000,
                    offset=next_offset,
                )
                ids.extend([p.id for p in points])
                if next_offset is None:
                    break
 
            if not ids:
                continue
 
            qdrant_client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=PointIdsList(points=ids),
            )
 
            total_deleted += len(ids)
            deleted_details.append({
                "filename": fname,
                "file_hash": files_map.get(fname, {}).get("file_hash"),
                "chunks_deleted": len(ids),
            })
 
        return {
            "source": source,
            "snapshot_file": path,
            "files_in_snapshot": len(files_map),
            "total_chunks_deleted": total_deleted,
            "deleted": deleted_details,
        }
 
    except Exception as e:
        logging.exception("sync_deleted failed")
        raise HTTPException(status_code=500, detail=f"sync_deleted failed: {e}")    
 
@app.post("/search")
def search_documents(search_in: SearchIn):
    query = search_in.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing text")

    # 1️⃣ Build filter (same as before)
    filter_conditions = []

    if search_in.source:
        filter_conditions.append(
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=search_in.source)
            )
        )

    if search_in.doc_title:
        filter_conditions.append(
            FieldCondition(
                key="metadata.doc_title",
                match=MatchValue(value=search_in.doc_title)
            )
        )

    search_filter = Filter(must=filter_conditions) if filter_conditions else None

    try:
        # 2️⃣ Embed query
        query_vector = embedding_model.embed_query(query)

        # 3️⃣ Latest Qdrant search
        points = qdrant_client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=search_in.k,
            with_payload=True,
            query_filter=search_filter,
            prefetch=[],
        ).points

        # 4️⃣ Convert → LangChain-style output
        matches = []

        for p in points:
            distance = p.score                    # cosine distance
            similarity = 1 - distance

            if similarity < search_in.threshold:
                continue

            metadata = p.payload.get("metadata", {})
            page_content = p.payload.get("page_content", {})

            # 🔁 Recreate LangChain Document
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )

            matches.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": round(similarity, 4),
                "distance": round(distance, 4),
            })

        return {
            "query": query,
            "results": matches,
            "total_results": len(matches),
            "threshold_used": search_in.threshold,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

 
@app.get("/count")
def count_documents(countin: CountIn):
    source = countin.source if countin.source else None
    scroll_filter = None
    if source:
        scroll_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source.strip()))])
 
    scroll_res, _ = qdrant_client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=10000
    )
    return {"count": len(scroll_res), "source": source or "all"}
 
@app.delete("/clear")
def clear_documents(source: Optional[str] = Query(None), filename: Optional[str] = Query(None)): # ⬅️ NEW
    try:
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
        if config.COLLECTION_NAME not in existing_collections:
            raise HTTPException(status_code=404, detail="Collection not found")
 
        must_conditions = []
 
        if source:
            must_conditions.append(
                FieldCondition(key="metadata.source", match=MatchValue(value=source.strip()))
            )
        if filename: # ⬅️ NEW
            must_conditions.append(
                FieldCondition(key="metadata.filename", match=MatchValue(value=filename.strip()))
            )
 
        scroll_filter = Filter(must=must_conditions) if must_conditions else None
 
        scroll_res, _ = qdrant_client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=10000
        )
 
        ids = [point.id for point in scroll_res]
        if not ids:
            return {
                "message": "No documents to delete",
                "source": source or "all",
                "filename": filename,
            }
 
        qdrant_client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=PointIdsList(points=ids)
        )
 
        return {
            "message": f"Deleted {len(ids)} documents",
            "source": source or "all",
            "filename": filename,
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
   
 
@app.get("/list")
def list_documents(source: Optional[str] = None):
   
    try:
 
        scroll_filter = None
        if source:
            scroll_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source.strip()))])
 
        scroll_res, _ = qdrant_client.scroll(
            collection_name=config.COLLECTION_NAME,
            with_payload=True,
            scroll_filter=scroll_filter,
            limit=10000
        )
 
        file_info = []
        for d in scroll_res:
            meta = d.payload.get("metadata", {})
            file_info.append(
                {
                    "doc_title": meta.get("doc_title", "unknown"),
                    "filename": meta.get("filename", "unknown"),
                    "creation_date": meta.get("creation_date", "unknown"),
                    "added_at": meta.get("added_at", "unknown"),
                    "source": meta.get("source", "unknown"),
                    "url": meta.get("url", "unknown"),
                    "file_hash": meta.get("file_hash", None),  # ⬅️ NEW
                }
            )
 
        # Remove duplicates
        seen = {}
        for f in file_info:
            if f["filename"] not in seen:
                seen[f["filename"]] = f
        file_info = list(seen.values())
 
        return {"source": source or "all", "file_count": len(file_info), "files": file_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/source")
def sources_list():
    try:
        scroll_res, _ = qdrant_client.scroll(
            collection_name=config.COLLECTION_NAME,
            with_payload=True,
            limit=10000,
        )
        sources = {d.payload.get("metadata", {}).get("source", "unknown") for d in scroll_res}
        return {"source_count": len(sources), "sources": list(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
 