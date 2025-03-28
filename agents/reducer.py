from .types import UserInfo

def user_preference_reducer(prev: UserInfo, update: UserInfo) -> UserInfo:
    if prev is None or prev == {}:
        return update
    if update is None or update == {}:
        return prev
    # delete any empty fields in the update
    update = {k: v for k, v in update.items() if v is not None}
    return {**prev, **update} # merge the two dictionaries