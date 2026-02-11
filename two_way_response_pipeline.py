from feedback_2d_cluster import RocchioFeedback2D
from interaction import elasticsearch_db

if __name__ == "__main__":

    rocchio_feedback_2d = RocchioFeedback2D(elasticsearch_db)

    project_id = "1770782030220"
    org_id = 1
    feedback_list = [
        (265619, -1),
        (182515, -1),
        (155086, 0),
        (128284, -1),
        (36960, -1),
        (11583, -1),
        (11299, 0),
        (1324, 0),
        (1304, 0),
        (1302, 0),
        (1402, 0),
        (1303, 0),
        (1258, 0),
        (1229, 0),
        (1205, 0),
        (1179, 0),
        (1177, 0),
        (1175, 0),
        (1147, 0),
        (1145, -1),
    ]

    rocchio_feedback_2d.run(
        query_vectors=None,
        project_id=project_id,
        org_id=org_id,
        embedding_index=None,
        feedback_list=feedback_list,
    )
