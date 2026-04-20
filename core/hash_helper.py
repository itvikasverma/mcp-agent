import hashlib

def compute_hash(content, algo="sha256"):
    """
    Convert string or bytes to a hash.
    """
    algo = algo.lower()
    if algo not in {"sha256", "sha512", "blake2b"}:
        raise ValueError("Unsupported algorithm.")

    hasher = getattr(hashlib, algo)() if algo != "blake2b" else hashlib.blake2b()

    if isinstance(content, str):
        content = content.encode("utf-8")
    elif not isinstance(content, (bytes, bytearray)):
        raise TypeError("Content must be str or bytes.")

    hasher.update(content)
    return hasher.hexdigest()
