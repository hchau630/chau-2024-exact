from matplotlib import rcParams

CM = 1 / 2.54  # cm to inch
CM_TO_PT = 28.3465  # cm to pt
AXSIZE = (2.1 * CM, 2.1 * CM)
CAXSIZE = (2.5 * CM, 2.1 * CM)
FIGSIZE = (3.6 * CM, 3.4 * CM)
CFIGSIZE = (4.3 * CM, 3.4 * CM)
ABSRECT = (0.95 * CM, 0.9 * CM, AXSIZE[0], AXSIZE[1])
RECT = (
    0.95 * CM / FIGSIZE[0],
    0.9 * CM / FIGSIZE[1],
    AXSIZE[0] / FIGSIZE[0],
    AXSIZE[1] / FIGSIZE[1],
)
CRECT = (
    0.95 * CM / CFIGSIZE[0],
    0.9 * CM / CFIGSIZE[1],
    CAXSIZE[0] / CFIGSIZE[0],
    CAXSIZE[1] / CFIGSIZE[1],
)
GREY = "#666666"


def get_sizes(leftscale=1.0, rightscale=1.0, topscale=1.0, bottomscale=1.0):
    leftextend = FIGSIZE[0] * (leftscale - 1)
    rightextend = FIGSIZE[0] * (rightscale - 1)
    topextend = FIGSIZE[1] * (topscale - 1)
    bottomextend = FIGSIZE[1] * (bottomscale - 1)
    figsize = (
        FIGSIZE[0] + leftextend + rightextend,
        FIGSIZE[1] + topextend + bottomextend,
    )
    rect = (
        (ABSRECT[0] + leftextend) / figsize[0],
        (ABSRECT[1] + bottomextend) / figsize[1],
        AXSIZE[0] / figsize[0],
        AXSIZE[1] / figsize[1],
    )
    return figsize, rect


def set_rcParams():
    rcParams["font.size"] = 7.25  # default: 10 pts
    rcParams["axes.labelpad"] = 0.0  # default: 4.0 pts
    rcParams["axes.titlepad"] = 4.0  # default: 6.0 pts
