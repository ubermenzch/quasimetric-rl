"""Unambiguous display labels for the two main result tables only.

Canonical environment names remain in source data and the task-goal table.
"""

TASK_LABELS = {
    'AntMaze UMaze-v5': 'AntMaze-U',
    'AntNavigate-v4': 'AntNavigate',
    'FetchPickAndPlace': 'FetchPnP',
    'FetchPush': 'FetchPush',
    'FetchSlide': 'FetchSlide',
    'maze2d-large': 'Maze2D-L',
    'PandaFlip-v3': 'PandaFlip',
    'PandaPickAndPlace-v3': 'PandaPnP',
    'PandaPickAndPlaceJoints-v3': 'PandaPnP-J',
    'PandaPushJoints-v3': 'PandaPush-J',
    'PandaSlide-v3': 'PandaSlide',
    'PointMaze Medium-v3': 'PointMaze-M',
    'Pusher-v4': 'Pusher',
    'reacher_hard': 'Reacher-H',
}


def task_label(name):
    return TASK_LABELS[name]
