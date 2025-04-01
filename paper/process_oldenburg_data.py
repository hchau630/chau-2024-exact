import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("paper/oldenburg_data/cell_table.csv")
    my_df = pd.read_pickle("paper/oldenburg_data/mine.pkl")
    stim_df = pd.read_pickle("paper/oldenburg_data/stims.pkl")
    cell_df = pd.read_pickle("paper/oldenburg_data/cells.pkl")
    assert isinstance(my_df, pd.DataFrame)
    assert isinstance(stim_df, pd.DataFrame)
    assert isinstance(cell_df, pd.DataFrame)
    print(df.dtypes)
    print(my_df.dtypes)
    print(df["ensPO"].unique())
    print(df["cellPO"].unique())
    print(stim_df["ens_pori"].unique())
    print(cell_df["pori"].unique())
    sf = df.query(
        "cellDist > 15 and offTarget == 0 and visP < 0.05 and cellOSI > 0.25 and cellEnsOSI > 0.7 and cellMeanEnsOSI > 0.5"
    )
    sf = sf.query("cellEnsMeaD > 200")
    # sf = sf.query("cellDist < 130")
    print(sf["cellEnsOSI"].unique())
    print(my_df["ens_osi"].unique())
    x = sf[["cellEnsOSI", "ensNum", "cellID", "cellOrisDiff", "dff"]]
    x = x.sort_values(["cellEnsOSI", "cellID"])
    x["cellOrisDiff"] = x["cellOrisDiff"].astype("Int64")
    x = x.reset_index(drop=True)

    y = my_df[
        [
            "ens_osi",
            "exp",
            "stim_id",
            "cell_idx",
            "delta_ens_pori_cell_pori",
            "delta_ens_pori_cell_pori_signed",
            "delta_r",
        ]
    ]
    y = y.sort_values(["ens_osi", "cell_idx"])
    y = y.rename(
        columns={
            "delta_ens_pori_cell_pori": "cellOrisDiff",
            "delta_r": "dff",
            "ens_oi": "cellEnsOSI",
            "cell_idx": "cellID",
        }
    )
    y = y.reset_index(drop=True)
    mask = x["cellOrisDiff"] != y["cellOrisDiff"]
    print(x[mask])
    print(y[mask])
    print(stim_df.query("exp == '191206_I138_1' and stim_id == 13"))
    print(cell_df.query("exp == '191206_I138_1' and cell_idx == 104")["pori"])
    # pd.testing.assert_frame_equal(
    #     x[["cellOrisDiff", "dff"]], y[["cellOrisDiff", "dff"]]
    # )

    sf = sf.groupby(["cellOrisDiff", "ensNum"], as_index=False)["dff"].mean()
    print(sf)
    # g = sns.relplot(sf, x="cellOrisDiff", y="dff", errorbar="se", kind="line")
    # g.refline(y=0)
    # # g.savefig("paper/figures/oldenburg/ori.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
