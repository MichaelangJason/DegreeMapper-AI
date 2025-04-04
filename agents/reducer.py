from typing import Dict, Any, List
from .types import UserInfo, Context, ContextUpdateDict
import logging

error_logger = logging.getLogger("uvicorn.error")

def user_info_reducer(prev: UserInfo, update: UserInfo) -> UserInfo:
    if prev is None or prev == {}:
        return update
    if update is None or update == {}:
        return prev
    # delete any empty fields in the update
    update = {k: v for k, v in update.items() if v is not None}
    return {**prev, **update} # merge the two dictionaries

def context_reducer(prev: Context, update: List[ContextUpdateDict]) -> Context:
    if update == []:
        return prev
    
    error_logger.error(update)
    
    next = {**prev}
    for item in update:
        op = item["op"]
        val = item["new_value"]
        type = item["type"]
        k = item["context_id"]

        if op == "update" and val is not None:
            next.update(
                { 
                    k: { 
                        "type": type,
                        "value": val
                    } 
                }
            )
        elif op == "delete":
            next.pop(k)
        elif op == "clear": # clears non user info contexts
            for k, v in prev.items():
                if v["type"] == "user_info": continue
                next.pop(k)
            break
        else:
            error_logger.error(op)
            error_logger.error(val)
            error_logger.error(item)
            raise ValueError("Operation value: ", op)

    return next
