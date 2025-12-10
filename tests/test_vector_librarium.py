from symbion.vector_librarium import VectorLibrarium
from symbion.latp_core import CoreSession


def test_vector_librarium_similarity():
    """VectorLibrarium should rank semantically closer texts higher."""
    lib = VectorLibrarium()

    cats = CoreSession(
        summary="Document about cats and dogs",
        main_theses=["Cats are small domestic animals"],
    )
    physics = CoreSession(
        summary="Notes on quantum field theory",
        main_theses=["Quantum fields and particles"],
    )

    cats_id = lib.store_core_session(cats)
    physics_id = lib.store_core_session(physics)

    results = lib.search_similar("cats and kittens", top_k=2)

    # We expect the "cats" document to come before "physics"
    returned_ids = [doc_id for doc_id, _ in results]

    assert cats_id in returned_ids
    assert returned_ids.index(cats_id) <= returned_ids.index(physics_id)
