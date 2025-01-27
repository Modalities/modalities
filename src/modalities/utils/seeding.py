import hashlib


def calculate_hashed_seed(input_data: list[str], max_seed: int = 2**32 - 1) -> int:
    # Calculate a seed from a list of strings
    # The seed is a number between 0 and max_seed (non-inclusive max_seed)
    def _hash_string(input_data: str) -> str:
        hash_object = hashlib.sha256(input_data.encode("utf-8"))
        hash_hex = hash_object.hexdigest()
        return hash_hex

    # even though this becomes an exremely large integer value,
    # we don't get overflows as python can represent integers of arbitrary size
    # https://docs.python.org/3/library/exceptions.html#OverflowError
    hash_strings = [_hash_string(x) for x in input_data]

    hash_sum = sum([int(x, 16) for x in hash_strings])
    print(hash_sum)
    print(type(hash_sum))

    seed = hash_sum % max_seed  # Ensure the seed fits within the max_seed range

    return seed
