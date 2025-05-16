import numpy as np


def main():
    # data from Figure S1A in Campagnola et al. (2022). Note that we need to take
    # the transpose of that data matrix so that the rows correspond to different
    # postsynaptic cell types while columns correspond to presynaptic cell types.
    L23_pmax = np.array(
        [
            [0.11, 0.65, 0.42, 0.11],
            [0.79, 0.79, 0.44, 0.04],
            [0.76, 0.28, 0.17, 0.49],
            [0.38, 0.0, 0.92, 0.05],
        ]
    )
    # print(L23_pmax.T)

    prob = np.array([640, 464, 1107])
    prob = prob / prob.sum()
    prob_sq = prob[:, None] * prob[None, :]
    np.testing.assert_allclose(prob.sum(), 1.0)
    np.testing.assert_allclose(prob_sq.sum(), 1.0)
    # print(prob)
    # print(prob_sq)

    EE = L23_pmax[0, 0]
    EI = (L23_pmax[0, 1:] * prob).sum()
    IE = (L23_pmax[1:, 0] * prob).sum()
    II = (L23_pmax[1:, 1:] * prob[:, None] * prob[None, :]).sum()
    L23_pmax = np.array(
        [
            [EE, EI],
            [IE, II],
        ]
    )
    print(f"L2/3 pmax =\n{L23_pmax}")


if __name__ == "__main__":
    main()
