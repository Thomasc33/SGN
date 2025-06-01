import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import plotly
from PIL import Image
from tqdm import tqdm



def render_frame(d):
    reshaped_data = d.reshape(-1, 3)
    x = reshaped_data[:, 0]
    y = reshaped_data[:, 1]
    z = reshaped_data[:, 2]

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.linspace(1, 25, len(x)),
                        color_continuous_scale='Rainbow', title='Interactive 3D Scatter Plot')

    fig.update_traces(marker=dict(size=2))

    cons = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]]

    for con in cons:
        lx = [x[con[0]], x[con[1]]]
        ly = [y[con[0]], y[con[1]]]
        lz = [z[con[0]], z[con[1]]]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2)))

    fig.show()

def render_video(d, gif=None, show_render=True, duration=100):
    cons = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]]

    frame_data = d[0].reshape(-1, 3)
    x = frame_data[:, 0]
    y = frame_data[:, 1]
    z = frame_data[:, 2]

    # Flatten the tensor to a 2D tensor with shape [frames * points, coordinates]
    d_flattened = d.reshape(-1, 3)

    # Calculate global bounds for x, y, and z
    x_min, x_max = d_flattened[:, 0].min().item(), d_flattened[:, 0].max().item()
    y_min, y_max = d_flattened[:, 1].min().item(), d_flattened[:, 1].max().item()
    z_min, z_max = d_flattened[:, 2].min().item(), d_flattened[:, 2].max().item()

    # Expand the bounds a bit for better visualization
    padding = 0.5
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    z_range = [z_min - padding, z_max + padding]

    # Set the fixed range for each axis
    scene = dict(
        xaxis=dict(range=x_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=y_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(range=z_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        camera=dict(
                eye=dict(x=0, y=0, z=-.9),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            ),
        aspectmode='cube',
        bgcolor='rgba(255,255,255,1)'
    )

    layout = go.Layout(updatemenus=[dict(type='buttons', showactive=False,
                                        buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])],
                    sliders=[dict(steps=[])],
                    title="Animated 3D Scatter Plot with Connections",
                    scene=scene,
                    autosize=False,
            )

    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                        marker=dict(size=2, color=np.linspace(1, 25, 25), colorscale='Rainbow'))

    traces = [scatter]

    for con in cons:
        lx = [x[con[0]], x[con[1]]]
        ly = [y[con[0]], y[con[1]]]
        lz = [z[con[0]], z[con[1]]]
        line_trace = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2))
        traces.append(line_trace)

    fig = go.Figure(data=traces, layout=layout)

    frame_list = []

    for i in range(d.shape[0]):
        frame_data = d[i].reshape(-1, 3)
        x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    
        fig.data[0].x = x
        fig.data[0].y = y
        fig.data[0].z = z

        frame_traces = []

        frame_scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=np.linspace(1, 25, 25), colorscale='Rainbow'))
        frame_traces.append(frame_scatter)

        for con in cons:
            lx = [x[con[0]], x[con[1]]]
            ly = [y[con[0]], y[con[1]]]
            lz = [z[con[0]], z[con[1]]]
            line_trace = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2))
            frame_traces.append(line_trace)

        frame = go.Frame(data=frame_traces, name=f'Frame {i}')
        frame_list.append(frame)

    fig.frames = frame_list

    if show_render: fig.show()

    if gif is not None:
        # Create a directory to save frames
        frame_dir = f'results/gif/{gif}'
        os.makedirs(frame_dir, exist_ok=True)

        layout = go.Layout(scene=scene, width=800, height=600, showlegend=False, margin=dict(l=0, r=0, b=0, t=0), autosize=False, )#paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Save each frame as an image
        for i in tqdm(range(len(frame_list))):
            frame = frame_list[i]
            fig = go.Figure(data=frame.data, layout=layout)
            plotly.io.write_image(fig, f'{frame_dir}/frame_{i}.png', width=800, height=600, scale=1)

        frames = glob.glob(f'{frame_dir}/frame_*.png')
        frames.sort()  # Ensure the frames are in order

        # Create an image object from the first frame
        img, *imgs = [Image.open(f) for f in frames]

        # Convert to GIF and save
        img.save(fp=f'results/gif/{gif}.gif', format='GIF', append_images=imgs,
                 save_all=True, duration=duration, loop=0)

        # Delete the frames after creating the GIF
        # for f in frames:
        #     os.remove(f)