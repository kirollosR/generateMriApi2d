def get_model(from_seq: str, to_seq: str):
    # Define your logic to map from sequence to model here
    model_mapping = {
        ("t1", "t1"): "model_for_t1_to_t1",
        ("t1", "t1ce"): "model_for_t1_to_t1ce",
        ("t1", "t2"): r"models/t1_to_t2",
        ("t1", "flair"): r"models/cyclegan_checkpoints.020.weights.h5",

        ("t1ce", "t1"): "model_for_t1ce_to_t1",
        ("t1ce", "t1ce"): "model_for_t1ce_to_t1ce",
        ("t1ce", "t2"): "model_for_t1ce_to_t2",
        ("t1ce", "flair"): "model_for_t1ce_to_flair",

        ("t2", "t1"): "model_for_t2_to_t1",
        ("t2", "t1ce"): "model_for_t2_to_t1ce",
        ("t2", "t2"): "model_for_t2_to_t2",
        ("t2", "flair"): "model_for_t2_to_flair",
    }
    return model_mapping.get((from_seq.lower(), to_seq.lower()), False)