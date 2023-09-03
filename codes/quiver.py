import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as anim, IPython, numpy as np, agentpy
from bokeh.plotting import figure, ColumnDataSource, save, show


def animation_plot(m, p):
    """
    This method takes in the model class and the parameters and returns an HTML display object.
    Parameters :
    - m : model class
    - p : parameter, a dictionary consisting of all the variables

    """
    projection = "3d" if p["ndim"] == 3 else None
    # plt.style.use("dark_background")
    fig = plt.figure(figsize=(7, 7))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection=projection)
    animation = agentpy.animate(m(p), fig, ax, animation_plot_single)
    # animation.save('scatter_3.gif',
    #      fps = 20, savefig_kwargs={'transparent': True, 'facecolor': 'none'}) #Uncomment for a no-axis, transparent visualisation
    return IPython.display.HTML(animation.to_jshtml(fps=20))


def animation_plot_single(m, ax):
    """
    This method takes in the model class and the previous axes and returns a single frame.
    Parameters :
    - m : model class
    - ax : axis

    """
    ndim = m.p.ndim
    ax.set_title(f"Boids Flocking Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.clear()
    ax.scatter(*pos, s=4, c="b", marker="o")
    ax.set_frame_on(False)
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()


def return_list(row):
    """
    This method is a dependency of scatter_position().

    """
    new_pos = []
    new_pos.append(row["pos_x"])
    new_pos.append(row["pos_y"])
    return new_pos


def scatter_position(df):

    """
    This method takes in the dataframe(that has already gone through update_df()) with columns 'pos_x' and 'pos_y' containing position values
    in X and Y coordinates respectively.

    Parameters :
    - df : a dataframe with columns 'pos_x' and 'pos_y'

    returns :
    - a scatter plot of pos_x vs pos_y at all timesteps

    dependency :
    - updated_dataframe(), return_list()
    """

    x_pos = list(df["pos_x"])
    y_pos = list(df["pos_y"])
    # output_file("line.html")

    df["position"] = df.apply(lambda row: return_list(row), axis=1)
    # print(df['position'])
    p = figure()
    # add a circle renderer with a size, color, and alpha
    p.circle(x_pos, y_pos, size=5, color="navy", alpha=0.5)
    show(p)


def quiver(grouped, param):

    x = grouped["pos_x"].loc[grouped["t"] == 1]  # of each agent
    y = grouped["pos_y"].loc[grouped["t"] == 1]
    u = grouped["vel_x"].loc[grouped["t"] == 1]
    v = grouped["vel_y"].loc[grouped["t"] == 1]

    fig = ff.create_quiver(
        x, y, u, v, scale=1.2, arrow_scale=1, line=dict(width=1.25, color="#8f180b")
    )

    fig.update_layout(
        width=50,
        height=50,
        title_text="Quiver Kinetics",
        title_x=0.5,
        xaxis_title="Size",
        xaxis_range=[0, param["size"]],
        yaxis_range=[0, param["size"]],
        yaxis_title="Size ",
    )

    frames = []
    for i in range(1, param["steps"]):
        figaux = ff.create_quiver(
            x=(grouped["pos_x"].loc[grouped["t"] == i]),
            y=((grouped["pos_y"].loc[grouped["t"] == i])),
            u=(grouped["vel_x"].loc[grouped["t"] == i]),
            v=(grouped["vel_y"].loc[grouped["t"] == i]),
            scale=1.2,
            arrow_scale=1,
        )

        frames.append(go.Frame(data=[figaux.data[0]]))

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=param["steps"], redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    fig.update(frames=frames)

    fig.show()
