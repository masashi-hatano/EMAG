from models.alignment import Alignment


def get_alignment(features_dim=512):
    alignment = Alignment(features_dim)
    return alignment
