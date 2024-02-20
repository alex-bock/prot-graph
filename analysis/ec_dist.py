
import sys

import pandas as pd
import plotly.express as px


if __name__ == "__main__":

    ec_df = pd.read_csv(sys.argv[1])
    ec_lvl = int(sys.argv[2])

    ec_df["ec_id"] = ec_df.apply(
        lambda x: ".".join([str(x[f"ec_{i}"]) for i in range(1, ec_lvl + 1)]),
        axis=1
    )

    fig = px.histogram(ec_df, x="ec_id")
    fig.update_xaxes(categoryorder="category ascending")
    fig.show()
