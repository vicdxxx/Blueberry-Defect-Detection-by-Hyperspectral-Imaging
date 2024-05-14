from spectral import *
from pathlib import Path
import cv2 as cv
import numpy as np
from spectral.io import envi
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

title_window = 'x'

data_dir = r'D:\Dataset\HyperspectralBlueberry\BlueberryScansforDestructiveTesting06132022'
data_dir = Path(data_dir)
good_dir = data_dir / 'GoodBlueberryScans'
bad_dir = data_dir / 'BadBlueberryScans'

data_dir1 = r'D:\Dataset\HyperspectralBlueberry\BlueberryScansforDestructiveTesting06142022'
data_dir1 = Path(data_dir1)
good_dir1 = data_dir1 / 'GoodBlueberryScans'
bad_dir1 = data_dir1 / 'BadBlueberryScans'
html_save_path = r"D:\test\xxx.html"
im_save_path = r"D:\test\xxx.svg"

load_spectral_mean_data_mean_dir = r'D:\Dataset\HyperspectralBlueberry\mean_data_python'

spatial_feature_type = 'tight_bbox'
save_dir_im_data_good = good_dir / spatial_feature_type / 'im_data_local'
save_dir_im_data_bad = bad_dir / spatial_feature_type / 'im_data_local'
save_dir_im_data_good1 = good_dir1 / spatial_feature_type / 'im_data_local'
save_dir_im_data_bad1 = bad_dir1 / spatial_feature_type / 'im_data_local'


def spli_mean_spectral_data(hyper_dict_good_tot, hyper_dict_bad_tot):
    hyper_dict_good_tot_0 = {}
    hyper_dict_good_tot_1 = {}
    for i_tot, key in enumerate(hyper_dict_good_tot):
        if i_tot < 84:
            hyper_dict_good_tot_0[key] = hyper_dict_good_tot[key]
        else:
            hyper_dict_good_tot_1[key] = hyper_dict_good_tot[key]

    hyper_dict_bad_tot_0 = {}
    hyper_dict_bad_tot_1 = {}
    for i_tot, key in enumerate(hyper_dict_bad_tot):
        if i_tot < 126:
            hyper_dict_bad_tot_0[key] = hyper_dict_bad_tot[key]
        else:
            hyper_dict_bad_tot_1[key] = hyper_dict_bad_tot[key]
    len(hyper_dict_good_tot_0), len(hyper_dict_bad_tot_0)
    return hyper_dict_good_tot_0, hyper_dict_good_tot_1, hyper_dict_bad_tot_0, hyper_dict_bad_tot_1


def load_im_dirs():
    im_dirs_good = []
    for dir_name in save_dir_im_data_good.iterdir():
        im_dirs_good.append(dir_name)

    for dir_name in save_dir_im_data_good1.iterdir():
        im_dirs_good.append(dir_name)
    im_dirs_good = sorted(im_dirs_good, key=lambda x: int(x.name))

    im_dirs_bad = []
    for dir_name in save_dir_im_data_bad.iterdir():
        im_dirs_bad.append(dir_name)

    for dir_name in save_dir_im_data_bad1.iterdir():
        im_dirs_bad.append(dir_name)
    im_dirs_bad = sorted(im_dirs_bad, key=lambda x: int(x.name))

    im_dirs = im_dirs_good + im_dirs_bad
    len(im_dirs)
    return im_dirs


def show_hyper_dict(hyper_dict, wavelengths, title='', type_idxes=None, alpha=1.0, names=None, size=15):
    one_sample_num = len(wavelengths)

    if type(hyper_dict) is dict:
        item_num = len(hyper_dict.keys())
        hyper_dict_sorted = sorted(hyper_dict.items(), key=lambda item: item[0])
    else:
        item_num = len(hyper_dict)
        hyper_dict_sorted = hyper_dict

    sku = []
    responce = []
    for idx, item in enumerate(hyper_dict_sorted):
        idx_name = None
        if len(item) == 2:
            idx_name, values = item
            print(idx_name)
        else:
            values = item.tolist()
            assert type_idxes is not None
        if idx_name is not None:
            sku += [idx_name]*one_sample_num
        elif type_idxes is None:
            sku += [str(idx+1)]*one_sample_num
        else:
            if names is not None:
                print('use names')
                sku += [names[idx]]*one_sample_num
            else:
                sku += [str(type_idxes[idx])]*one_sample_num
        responce += values

    x = wavelengths*item_num
    df = pd.DataFrame(dict(
        wavelength=x,
        reflectance=responce,
        label=sku,
    ))
    color = ('rgba('+str(np.random.randint(0, high=256))+',' +
             str(np.random.randint(0, high=256))+',' +
             str(np.random.randint(0, high=256)))
    
    fig = px.line(df, x="wavelength", y="reflectance", color='label', 
                  labels={
                        "wavelength": "Wavelength (nm)",
                        "reflectance": "Reflectance (a.u.)",
                  },
                  )
    # import plotly.graph_objects as go
    # fig.add_trace(go.Scatter(x=fig['data'][1].x, y=fig['data'][1].y, fill='tozeroy', line=dict(color='rgb(155, 38, 0)'),
    #                     mode='none' # override default markers+lines
    #                     ))
    # fig.add_trace(go.Scatter(x=fig['data'][2].x, y=fig['data'][2].y, fill='tozeroy', line=dict(color='rgb(255, 255, 255)'),
    #                     mode='none' # override default markers+lines
    #                     ))
    if len(names) != 0:
        # newnames = names
        # fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
        #                                 legendgroup = newnames[t.name],
        #                                 hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
        for idx_name, name in enumerate(names):
            fig['data'][idx_name]['showlegend']=True
            fig['data'][idx_name]['name']=name
    
    #fig.for_each_trace(lambda t: t.update(name=t.name,))
    #fig.update_layout(showlegend=True)

    if len(fig['data']) == 6:
        fig['data'][0]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][1]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][2]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][1]['line']['dash'] = 'dash'
        fig['data'][2]['line']['dash'] = 'dash'

        fig['data'][3]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][4]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][5]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][4]['line']['dash'] = 'dash'
        fig['data'][5]['line']['dash'] = 'dash'

    if 1:
        # fig.update_layout({
        #    'plot_bgcolor': 'rgba(255,255,255,1)',
        #    'paper_bgcolor': 'rgba(255,255,255,1)'
        # })
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                zeroline=False,
                # linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Times New Roman',
                    size=size,
                    color='rgb(0, 0, 0, 1)',
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                zeroline=False,
                # linecolor='rgb(204, 204, 204)',
                ticks='inside',
                tickfont=dict(
                    family='Times New Roman',
                    size=size,
                    color='rgb(0, 0, 0, 1)',
                ),
            ),
            autosize=False,
            # automargin=True,
            #margin=dict(
            #    autoexpand=False,
            #    l=50,
            #    r=50,
            #    t=30,
            #    b=50,
            #),
            #margin=dict(
            #    autoexpand=False,
            #    l=60,
            #    r=60,
            #    t=40,
            #    b=60,
            #),
            #margin=dict(
            #    autoexpand=False,
            #    l=85,
            #    r=30,
            #    t=20,
            #    b=80,
            #),
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        fig.update_xaxes(range=[wavelengths[0], wavelengths[-1]])
        fig.update_xaxes(ticks="outside")
        fig.update_yaxes(ticks="inside")
        #fig.update_layout(title_text='<b>'+title+'</b>', title_x=0.5, title_font_family="Times New Roman",)
        #fig.update_layout(title_text=title, title_x=0.5, title_font_family="Times New Roman",)
        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="black",
            legend_title_font_color="black",
            font_size=size
        )

        fig.add_shape(
            # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="black",
                width=1,
            )
        )

    fig.write_html(html_save_path)
    print(html_save_path)
    
    # dpi = 800
    # width_in_mm = 130
    # width_default_px = 800
    # # scale = (width_in_mm / 25.4) / (width_default_px / dpi) = (130 / 25.4) / (700 / 300) = 2.1934758155230596
    # scale = (width_in_mm / 25.4) / (width_default_px / dpi)
    # fig.write_image(im_save_path, scale=scale)
    print(im_save_path)


def show_hyper_dict_mean_std(hyper_dict, wavelengths, title='', type_idxes=None, alpha=1.0, names=None, size=28):
    one_sample_num = len(wavelengths)

    if type(hyper_dict) is dict:
        item_num = len(hyper_dict.keys())
        hyper_dict_sorted = sorted(hyper_dict.items(), key=lambda item: item[0])
    else:
        item_num = len(hyper_dict)
        hyper_dict_sorted = hyper_dict

    sku = []
    responce = []
    for idx, item in enumerate(hyper_dict_sorted):
        idx_name = None
        if len(item) == 2:
            idx_name, values = item
            print(idx_name)
        else:
            values = item.tolist()
            assert type_idxes is not None
        if idx_name is not None:
            sku += [idx_name]*one_sample_num
        elif type_idxes is None:
            sku += [str(idx+1)]*one_sample_num
        else:
            if names is not None:
                print('use names')
                sku += [names[idx]]*one_sample_num
            else:
                sku += [str(type_idxes[idx])]*one_sample_num
        responce += values

    x = wavelengths*item_num
    df = pd.DataFrame(dict(
        wavelength=x,
        reflectance=responce,
        label=sku,
    ))
    color = ('rgba('+str(np.random.randint(0, high=256))+',' +
             str(np.random.randint(0, high=256))+',' +
             str(np.random.randint(0, high=256)))
    
    fig = px.line(df, x="wavelength", y="reflectance", color='label', 
                  labels={
                        "wavelength": "Wavelength (nm)",
                        "reflectance": "Reflectance (a.u.)",
                  },
                  )

    for idx in range(6):
        fig.data[idx].visible=False

    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(
    x=np.concatenate([fig['data'][2].x, fig['data'][1].x[::-1]]),
    y=np.concatenate([fig['data'][2].y, fig['data'][1].y[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.1)',
    marker_color='rgba(255, 165, 0, 0.9)',
    hoveron='points',
    name='Std Region of the Sound',
    line = dict(  dash='dash')
    ))

    fig.add_trace(go.Scatter(
    x=np.concatenate([fig['data'][0].x]),
    y=np.concatenate([fig['data'][0].y]),
    marker_color='rgba(255, 165, 0, 1.0)',
    hoveron='points',
    name='Mean of the Sound',
    ))

    fig.add_trace(go.Scatter(
    x=np.concatenate([fig['data'][5].x, fig['data'][4].x[::-1]]),
    y=np.concatenate([fig['data'][5].y, fig['data'][4].y[::-1]]),
    fill='toself',
    fillcolor='rgba(133, 198, 8, 0.1)',
    marker_color='rgba(133, 198, 8, 0.9)',
    hoveron='points',
    name='Std Region of the Defective',
    line = dict( dash='dash')
    ))

    fig.add_trace(go.Scatter(
    x=np.concatenate([fig['data'][3].x]),
    y=np.concatenate([fig['data'][3].y]),
    marker_color='rgba(133, 198, 8, 1.0)',
    hoveron='points',
    name='Mean of the Defective',
    ))
    if len(names) != 0:
        for idx_name, name in enumerate(names):
            fig['data'][idx_name]['showlegend']=True
            fig['data'][idx_name]['name']=name
    
    if len(fig['data']) == 6:
        fig['data'][0]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][1]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][2]['line']['color'] = 'rgb(255, 138, 0)'
        fig['data'][1]['line']['dash'] = 'dash'
        fig['data'][2]['line']['dash'] = 'dash'

        fig['data'][3]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][4]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][5]['line']['color'] = 'rgb(133, 198, 89)'
        fig['data'][4]['line']['dash'] = 'dash'
        fig['data'][5]['line']['dash'] = 'dash'

    if 1:
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                zeroline=False,
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Times New Roman',
                    size=size+2,
                    color='rgb(0, 0, 0, 1)',
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                zeroline=False,
                ticks='inside',
                tickfont=dict(
                    family='Times New Roman',
                    size=size+2,
                    color='rgb(0, 0, 0, 1)',
                ),
            ),
            autosize=False,
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white',

            margin=dict(
                autoexpand=False,
                l=85,
                r=30,
                t=20,
                b=80,
            ),
        )
        fig.update_xaxes(range=[wavelengths[0], wavelengths[-1]])
        fig.update_xaxes(ticks="outside")
        fig.update_yaxes(ticks="inside")
        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="black",
            legend_title_font_color="black",
            font_size=size-4
        )

        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="black",
                width=1,
            )
        )
    fig.update_layout(legend_title_text='', legend=dict(bgcolor='rgba(0,0,0,0)', font = dict(family = "Times New Roman", size = size-4), yanchor="top", y=0.98, xanchor="left", x=0.02))
    fig.write_html(html_save_path)
    print(html_save_path)
    
    dpi = 800
    width_in_mm = 130
    width_default_px = 800
    # scale = (width_in_mm / 25.4) / (width_default_px / dpi) = (130 / 25.4) / (700 / 300) = 2.1934758155230596
    scale = (width_in_mm / 25.4) / (width_default_px / dpi)
    fig.write_image(im_save_path, scale=scale)
    print(im_save_path)


def show_single_spectral_im_pixel_distribution(im_corrected_datas, name=None):
    """
    this cell shows distribution, but need 8s to run
    """
    import numpy as np
    import plotly as py
    import plotly.graph_objs as go
    import pandas as pd
    if name is None:
        name = 'part'
    for i_im, diff_im in enumerate(im_corrected_datas):
        print(name+'_'+str(i_im), diff_im.min(), diff_im.max())

    type0 = []
    intensity0 = []
    for i_im, im_corrected_data in enumerate(im_corrected_datas):
        tmp = im_corrected_data.reshape(-1)
        data_num = tmp.shape[0]
        type0 += [str(i_im)]*data_num
        intensity0 += tmp.tolist()
    df = pd.DataFrame(dict(
        intensity=intensity0,
        type=type0,
    ))
    fig = px.histogram(df, x="intensity", color='type')
    fig.write_html(html_save_path)
    print(html_save_path)


def find_target_bands(target_wavelengths, data):
    # target_wavelengths = [640, 550, 460]
    # target_wavelengths = [800, 810, 820]

    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]

    target_values = []
    target_idxes = []
    for target in target_wavelengths:
        nearest_value, nearest_idx = find_nearest(wavelengths, target)
        target_values.append(nearest_value)
        target_idxes.append(nearest_idx)
    return target_values, target_idxes


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_calibrated_image(target_wavelength, wavelengths, data, ref_data, show=False):
    band, idx = find_nearest(wavelengths, target_wavelength)
    print(band, idx)
    idx = int(idx)
    im = data[:, :, idx]
    ref = ref_data[:data.shape[0], :, idx]
    im = im / ref
    if show:
        im = (im-im.min()) / (im.max()-im.min())
        im_np = np.array(im*255, dtype=np.uint8).squeeze()
        fig = px.imshow(im_np)
        fig.show()
    return im


def im_norm_255(im):
    im = (im-im.min()) / (im.max()-im.min())
    im_np = np.array(im*255, dtype=np.uint8).squeeze()
    return im_np


def show_im(im, norm=True):
    if norm:
        im_np = im_norm_255(im)
        fig = px.imshow(im_np)
        fig.show()
    else:
        fig = plt.imshow(im)
        plt.show()


def show_hyper_spectral_data(data, ref_data, wavelengths, title_window=title_window):
    def on_trackbar(idx):
        im = data[:, :, idx]
        ref = ref_data[:, :, idx]
        im_corrected = im/ref
        im_corrected = im_corrected/im_corrected.max()
        print("\r", f'wavelengths: {wavelengths[idx]}', end="", flush=True)
        cv.imshow(title_window, im_corrected)
    cv.namedWindow(title_window, 0)
    slider_max = data.shape[2]
    trackbar_name = f'{wavelengths[0]}:{wavelengths[-1]}'
    cv.createTrackbar(trackbar_name, title_window, 0, slider_max, on_trackbar)
    on_trackbar(0)
    cv.waitKey()


def find_hyper_spectral_data_range(data, ref_data):
    band_num = data.shape[2]
    im_mins = []
    im_maxs = []
    ref_mins = []
    ref_maxs = []
    final_mins = []
    final_maxs = []
    for idx in range(band_num):
        im = data[:, :, idx]
        ref = ref_data[:, :, idx]
        im_mins.append(im.min())
        im_maxs.append(im.max())

        ref_mins.append(ref.min())
        ref_maxs.append(ref.max())

        corrected = im / ref
        final_mins.append(corrected.min())
        final_maxs.append(corrected.max())

    im_mins = np.array(im_mins)
    im_maxs = np.array(im_maxs)
    ref_mins = np.array(ref_mins)
    ref_maxs = np.array(ref_maxs)
    final_mins = np.array(final_mins)
    final_maxs = np.array(final_maxs)


def show_hyper_spectral_data_range(data, im_mins, im_maxs, ref_mins, ref_maxs, final_mins, final_maxs):
    data_num = im_maxs.shape[0]
    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]
    x_data = np.array(wavelengths)

    type = ['im_mins']*data_num+['im_maxs']*data_num+['ref_mins']*data_num+['ref_maxs']*data_num+['final_mins']*data_num+['final_maxs']*data_num
    x = x_data.tolist()*6

    df = pd.DataFrame(dict(
        band=x,
        responce=im_mins.tolist() + im_maxs.tolist() + ref_mins.tolist() + ref_maxs.tolist() + final_mins.tolist() + final_maxs.tolist(),
        type=type,
    ))
    fig = px.line(df, x="band", y="responce", color='type')
    fig.write_html(html_save_path)
    print(html_save_path)


def show_mean_spectral_data(hyper_dict, idx, wavelengths):
    key = list(hyper_dict.keys())[idx]
    y = hyper_dict[key]
    x = wavelengths
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y
    ))
    fig.show()


def show_single_spectral(data, idx):
    im = data[:, :, idx]
    im_np = np.array(im/im.max()*255, dtype=np.uint8).squeeze()
    print(im_np.shape)
    fig = px.imshow(im_np)
    fig.show()


def load_hyper_spectral_data():
    ref_names = ['WhiteReference']
    ref_path = data_dir / f'{ref_names[0]}.bil'.__str__()
    ref_path_head = data_dir / f'{ref_names[0]}.bil.hdr'.__str__()
    ref_data = envi.open(ref_path_head, ref_path)

    names = ['Good Blueberry 1-42', 'Good Blueberry 43-84']
    path = good_dir / f'{names[0]}.bil'.__str__()
    path_head = good_dir / f'{names[0]}.bil.hdr'.__str__()
    path1 = good_dir / f'{names[1]}.bil'.__str__()
    path_head1 = good_dir / f'{names[1]}.bil.hdr'.__str__()
    data_good = envi.open(path_head, path)
    data_good1 = envi.open(path_head1, path1)
    print(data_good)

    names = ['Bad Blueberry 1-42', 'Bad Blueberry 43-84', 'Bad Blueberry 85-126']
    ref_names = ['WhiteReference']
    path = bad_dir / f'{names[0]}.bil'.__str__()
    path_head = bad_dir / f'{names[0]}.bil.hdr'.__str__()
    path1 = bad_dir / f'{names[1]}.bil'.__str__()
    path_head1 = bad_dir / f'{names[1]}.bil.hdr'.__str__()
    path2 = bad_dir / f'{names[2]}.bil'.__str__()
    path_head2 = bad_dir / f'{names[2]}.bil.hdr'.__str__()
    data_bad = envi.open(path_head, path)
    data_bad1 = envi.open(path_head1, path1)
    data_bad2 = envi.open(path_head2, path2)
    print(data_bad)

    names = ['Good Blueberry 85-126', 'Good Blueberry 127-168', 'Good Blueberry 169-210']
    path = good_dir1 / f'{names[0]}.bil'.__str__()
    path_head = good_dir1 / f'{names[0]}.bil.hdr'.__str__()
    path1 = good_dir1 / f'{names[1]}.bil'.__str__()
    path_head1 = good_dir1 / f'{names[1]}.bil.hdr'.__str__()
    path2 = good_dir1 / f'{names[2]}.bil'.__str__()
    path_head2 = good_dir1 / f'{names[2]}.bil.hdr'.__str__()
    data_good2 = envi.open(path_head, path)
    data_good3 = envi.open(path_head1, path1)
    data_good4 = envi.open(path_head2, path2)

    names = ['Bad Blueberries 127-168', 'Bad Blueberries 169-210']
    ref_names = ['WhiteReference']
    path = bad_dir1 / f'{names[0]}.bil'.__str__()
    path_head = bad_dir1 / f'{names[0]}.bil.hdr'.__str__()
    path1 = bad_dir1 / f'{names[1]}.bil'.__str__()
    path_head1 = bad_dir1 / f'{names[1]}.bil.hdr'.__str__()
    data_bad3 = envi.open(path_head, path)
    data_bad4 = envi.open(path_head1, path1)

    hyper_spectral_data_dict = {}
    hyper_spectral_data_dict['ref_data'] = ref_data
    hyper_spectral_data_dict['data_good'] = data_good
    hyper_spectral_data_dict['data_good1'] = data_good1
    hyper_spectral_data_dict['data_bad'] = data_bad
    hyper_spectral_data_dict['data_bad1'] = data_bad1
    hyper_spectral_data_dict['data_bad2'] = data_bad2

    hyper_spectral_data_dict['data_good2'] = data_good2
    hyper_spectral_data_dict['data_good3'] = data_good3
    hyper_spectral_data_dict['data_good4'] = data_good4
    hyper_spectral_data_dict['data_bad3'] = data_bad3
    hyper_spectral_data_dict['data_bad4'] = data_bad4

    data_idx_dict = {}
    data_idx_dict['data_good'] = {'data_dir': good_dir, 'data': data_good, 'start_idx': 0, 'name': 'good_1_42', 'crop_x_start': 170, 'crop_y_start': 50}
    data_idx_dict['data_good1'] = {'data_dir': good_dir, 'data': data_good1, 'start_idx': 42, 'name': 'good_43_84', 'crop_x_start': 170, 'crop_y_start': 50}
    data_idx_dict['data_bad'] = {'data_dir': bad_dir, 'data': data_bad, 'start_idx': 0, 'name': 'bad_1_42', 'crop_x_start': 170, 'crop_y_start': 50}
    data_idx_dict['data_bad1'] = {'data_dir': bad_dir, 'data': data_bad1, 'start_idx': 42, 'name': 'bad_43_84', 'crop_x_start': 170, 'crop_y_start': 50}
    data_idx_dict['data_bad2'] = {'data_dir': bad_dir, 'data': data_bad2, 'start_idx': 84, 'name': 'bad_85_126', 'crop_x_start': 170, 'crop_y_start': 50}

    data_idx_dict['data_good2'] = {'data_dir': good_dir1, 'data': data_good2, 'start_idx': 84, 'name': 'good_85_126'}
    data_idx_dict['data_good3'] = {'data_dir': good_dir1, 'data': data_good3, 'start_idx': 126, 'name': 'good_127_168'}
    data_idx_dict['data_good4'] = {'data_dir': good_dir1, 'data': data_good4, 'start_idx': 168, 'name': 'good_169_210'}
    data_idx_dict['data_bad3'] = {'data_dir': bad_dir1, 'data': data_bad3, 'start_idx': 126, 'name': 'bad_127_168'}
    data_idx_dict['data_bad4'] = {'data_dir': bad_dir1, 'data': data_bad4, 'start_idx': 168, 'name': 'bad_169_210'}
    return hyper_spectral_data_dict, data_idx_dict


def load_spectral_mean_data(use_part_1=True, use_part_2=True, norm_spectral_feature=False):
    # save_path_good = good_dir / 'good_1_42.npy'.__str__()
    # save_path_good1 = good_dir / 'good_43_84.npy'.__str__()
    # save_path_bad = bad_dir / 'bad_1_42.npy'.__str__()
    # save_path_bad1 = bad_dir / 'bad_43_84.npy'.__str__()
    # save_path_bad2 = bad_dir / 'bad_85_126.npy'.__str__()

    # hyper_dict_good = np.load(save_path_good, allow_pickle='TRUE').item()
    # hyper_dict_good1 = np.load(save_path_good1, allow_pickle='TRUE').item()
    # hyper_dict_bad = np.load(save_path_bad, allow_pickle='TRUE').item()
    # hyper_dict_bad1 = np.load(save_path_bad1, allow_pickle='TRUE').item()
    # hyper_dict_bad2 = np.load(save_path_bad2, allow_pickle='TRUE').item()

    # hyper_dict_new = {}
    # for key in hyper_dict_good:
    #    hyper_dict_new[key] = hyper_dict_good[key]
    # for key in hyper_dict_good1:
    #    hyper_dict_new[key+42] = hyper_dict_good1[key]
    # hyper_dict_new.keys()
    # hyper_dict_good_tot = hyper_dict_new

    # hyper_dict_new = {}
    # for key in hyper_dict_bad:
    #    hyper_dict_new[key] = hyper_dict_bad[key]
    # for key in hyper_dict_bad1:
    #    hyper_dict_new[key+42] = hyper_dict_bad1[key]
    # for key in hyper_dict_bad2:
    #    hyper_dict_new[key+84] = hyper_dict_bad2[key]
    # hyper_dict_new.keys()
    # hyper_dict_bad_tot = hyper_dict_new
    global load_spectral_mean_data_mean_dir
    mean_dir = load_spectral_mean_data_mean_dir
    mean_dir = Path(mean_dir)
    if use_part_1:
        save_path_good = mean_dir / 'good_1_42.npy'.__str__()
        save_path_good1 = mean_dir / 'good_43_84.npy'.__str__()
        save_path_bad = mean_dir / 'bad_1_42.npy'.__str__()
        save_path_bad1 = mean_dir / 'bad_43_84.npy'.__str__()
        save_path_bad2 = mean_dir / 'bad_85_126.npy'.__str__()

        hyper_dict_good = np.load(save_path_good, allow_pickle='TRUE').item()
        hyper_dict_good1 = np.load(save_path_good1, allow_pickle='TRUE').item()
        hyper_dict_bad = np.load(save_path_bad, allow_pickle='TRUE').item()
        hyper_dict_bad1 = np.load(save_path_bad1, allow_pickle='TRUE').item()
        hyper_dict_bad2 = np.load(save_path_bad2, allow_pickle='TRUE').item()
    if use_part_2:
        save_path_good2 = mean_dir / 'good_85_126.npy'.__str__()
        save_path_good3 = mean_dir / 'good_127_168.npy'.__str__()
        save_path_good4 = mean_dir / 'good_169_210.npy'.__str__()
        save_path_bad3 = mean_dir / 'bad_127_168.npy'.__str__()
        save_path_bad4 = mean_dir / 'bad_169_210.npy'.__str__()

        hyper_dict_good2 = np.load(save_path_good2, allow_pickle='TRUE').item()
        hyper_dict_good3 = np.load(save_path_good3, allow_pickle='TRUE').item()
        hyper_dict_good4 = np.load(save_path_good4, allow_pickle='TRUE').item()
        hyper_dict_bad3 = np.load(save_path_bad3, allow_pickle='TRUE').item()
        hyper_dict_bad4 = np.load(save_path_bad4, allow_pickle='TRUE').item()

    hyper_dict_new = {}
    if use_part_1:
        for key in hyper_dict_good:
            hyper_dict_new[key] = hyper_dict_good[key]
        for key in hyper_dict_good1:
            hyper_dict_new[key+42] = hyper_dict_good1[key]
    if use_part_2:
        for key in hyper_dict_good2:
            hyper_dict_new[key+84] = hyper_dict_good2[key]
        for key in hyper_dict_good3:
            hyper_dict_new[key+126] = hyper_dict_good3[key]
        for key in hyper_dict_good4:
            hyper_dict_new[key+168] = hyper_dict_good4[key]
    hyper_dict_new.keys()
    hyper_dict_good_tot = hyper_dict_new

    hyper_dict_new = {}
    if use_part_1:
        for key in hyper_dict_bad:
            hyper_dict_new[key] = hyper_dict_bad[key]
        for key in hyper_dict_bad1:
            hyper_dict_new[key+42] = hyper_dict_bad1[key]
        for key in hyper_dict_bad2:
            hyper_dict_new[key+84] = hyper_dict_bad2[key]
    if use_part_2:
        for key in hyper_dict_bad3:
            hyper_dict_new[key+126] = hyper_dict_bad3[key]
        for key in hyper_dict_bad4:
            hyper_dict_new[key+168] = hyper_dict_bad4[key]
    hyper_dict_new.keys()
    hyper_dict_bad_tot = hyper_dict_new

    if norm_spectral_feature:
        """
        for convenient, do not split train / val / test dataset here, use all dataset to normalize
        in practice, should only use train dataset for normalizing
        then use mean_train, std_train to norm val / test dataset
        """
        good_keys = list(hyper_dict_good_tot.keys())
        bad_keys = list(hyper_dict_bad_tot.keys())
        key_sample = good_keys[0]
        band_num = len(hyper_dict_good_tot[key_sample])
        band_means = []
        band_stds = []
        for i_band in range(band_num):
            band_samples = []
            for good_key in good_keys:
                band_sample = hyper_dict_good_tot[good_key][i_band]
                band_samples.append(band_sample)
            for bad_key in bad_keys:
                band_sample = hyper_dict_bad_tot[bad_key][i_band]
                band_samples.append(band_sample)
            band_samples = np.array(band_samples)
            # print(i_band, band_samples.mean(), band_samples.std())
            band_means.append(band_samples.mean())
            band_stds.append(band_samples.std())
        for i_band in range(band_num):
            all_samples = []
            for good_key in good_keys:
                tmp = (hyper_dict_good_tot[good_key][i_band] - band_means[i_band]) / band_stds[i_band]
                hyper_dict_good_tot[good_key][i_band] = tmp
                all_samples.append(tmp)
            for bad_key in bad_keys:
                tmp = (hyper_dict_bad_tot[bad_key][i_band] - band_means[i_band]) / band_stds[i_band]
                hyper_dict_bad_tot[bad_key][i_band] = tmp
                all_samples.append(tmp)
            all_samples = np.array(all_samples)
            # print(i_band, all_samples.mean(), all_samples.std())
    return hyper_dict_good_tot, hyper_dict_bad_tot


def load_spectral_mean_data_xy(use_part_1=True, use_part_2=True, good_keys=None, bad_keys=None, return_dict=False):
    hyper_dict_good_tot, hyper_dict_bad_tot = load_spectral_mean_data(use_part_1, use_part_2)
    hyper_dict_tot = {}
    x = []
    y = []
    if good_keys is not None:
        loop_var = good_keys
    else:
        loop_var = hyper_dict_good_tot
    offset = len(loop_var)
    for i_good, key in enumerate(loop_var):
        x.append(hyper_dict_good_tot[key])
        y.append(0)
        hyper_dict_tot[key] = hyper_dict_good_tot[key]

    if bad_keys is not None:
        loop_var = bad_keys
    else:
        loop_var = hyper_dict_bad_tot
    for i_bad, key in enumerate(loop_var):
        x.append(hyper_dict_bad_tot[key])
        y.append(1)
        hyper_dict_tot[key+offset] = hyper_dict_bad_tot[key]

    x = np.array(x)
    y = np.array(y)
    if return_dict:
        return x, y, hyper_dict_tot
    else:
        return x, y


def load_spectral_mean_data_xy_matlab(dir):
    from scipy.io import loadmat
    names = ['badBerry1.mat', 'badBerry2.mat', 'badBerry3.mat']
    import os

    x = []
    y = [1] * 126 + [0] * 84
    for name in names:
        print(name)
        path = dir+name
        data = loadmat(path)
        x.append(data['tempSpectraM'])
        print(data['tempSpectraM'].shape)

    names = ['goodBerry1.mat', 'goodBerry1.mat']
    for name in names:
        print(name)
        path = dir+name
        data = loadmat(path)
        x.append(data['berrySpectraM'])
        print(data['berrySpectraM'].shape)

    x = np.array(x).reshape(-1, 462)
    y = np.array(y)
    return x, y


def standard_pipeline_get_contours(ref_data, data_idx_dict, data_name, show=False):
    data = data_idx_dict[data_name]['data']
    sample_start_idx = data_idx_dict[data_name]['start_idx']
    save_name = data_idx_dict[data_name]['name']

    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]
    data_num = len(wavelengths)
    title_window = 'x'

    im1 = get_calibrated_image(680, wavelengths, data, ref_data)
    im2 = get_calibrated_image(900, wavelengths, data, ref_data)

    diff_im = im2-im1
    diff_im = diff_im[:, :, 0]

    idxes = np.where(diff_im < 0.0)
    diff_im_optimize = diff_im.copy()
    diff_im_optimize[idxes[0], idxes[1]] = 0

    diff_im_optimize = diff_im_optimize / diff_im_optimize.max()
    diff_im_optimize = np.round(diff_im_optimize*255).astype(np.uint8)

    diff_im_optimize_eh = cv.equalizeHist(diff_im_optimize)

    from skimage.filters import threshold_yen, threshold_otsu, try_all_threshold
    thresh = threshold_yen(diff_im_optimize_eh)
    ret, diff_im_optimize_t = cv.threshold(diff_im_optimize_eh, thresh-50, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    iterations = 1
    eroded = cv.erode(diff_im_optimize_t, kernel, iterations=iterations)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    iterations = 4
    diff_im_optimize_open = cv.dilate(eroded, kernel, iterations=iterations)

    iterations = 8
    diff_im_optimize_open = cv.dilate(diff_im_optimize_open, kernel, iterations=iterations)
    diff_im_optimize_open = cv.erode(diff_im_optimize_open, kernel, iterations=iterations)

    iterations = 4
    diff_im_optimize_open = cv.erode(diff_im_optimize_open, kernel, iterations=iterations)

    from modeling_filter_metal_plate import detect_circle
    find_circle_list, img_bgr = detect_circle(diff_im_optimize_open)

    res = cv.findContours(image=diff_im_optimize_open, mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = res
    valid_contours = []
    im_circles = np.zeros_like(diff_im_optimize_open)

    for circle in find_circle_list:
        x, y, r = circle
        cv.circle(im_circles, (x, y), r, 255, -1)

    for contour in contours:
        im_one_conotur = np.zeros_like(diff_im_optimize_open)
        cv.drawContours(image=im_one_conotur, contours=[contour], contourIdx=-1,
                        color=255, thickness=-1, lineType=cv.LINE_AA)
        res_im = im_circles * im_one_conotur
        idxes = np.where(res_im > 0)
        if len(idxes[0]) > 0:
            valid_contours.append(contour)
    contours = valid_contours

    diff_3_channels = diff_im_optimize[:, :, None]
    diff_3_channels = np.concatenate([diff_3_channels, diff_3_channels, diff_3_channels], 2).squeeze()

    im_copy = diff_3_channels.copy()
    cv.drawContours(image=im_copy, contours=contours, contourIdx=-1,
                    color=[10, 200, 50], thickness=2, lineType=cv.LINE_AA)

    font = cv.FONT_HERSHEY_DUPLEX
    center_pts = []
    center_refs = []

    orders = []
    for i_contour, contour in enumerate(contours):
        x = int(contour[:, :, 0].mean())
        y = int(contour[:, :, 1].mean())
        center_refs.append(x+7*y)
        center_pts.append((x, y))
    idx_sorted = np.argsort(center_refs)
    for i_idx_sorted, idx in enumerate(idx_sorted):
        x, y = center_pts[idx]
        order = i_idx_sorted + 1 + sample_start_idx
        orders.append(order)
        cv.putText(im_copy, str(order), (x-5, y+5), font, 1, [10, 50, 200], 2)
    print('contours num:', len(contours))

    orders = np.array(orders)
    gap = orders[1:] - orders[:-1]
    idxes = np.where(gap != 1)
    if len(idxes[0]) > 0:
        print(orders[idxes[0]])
    else:
        print('order OK')
    assert len(idxes[0]) == 0
    if show:
        cv.namedWindow(title_window, 0)
        cv.imshow(title_window, im_copy)
        res = cv.waitKey(0)
    return data, idx_sorted, contours


def get_mean_spectral_data(data, ref_data, idx_sorted, contours):
    import cv2 as cv
    from tqdm import tqdm

    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]
    data_num = len(wavelengths)
    hyper_dict = {}
    for i_band in tqdm(range(data_num), ncols=80):
        im = data[:, :, i_band]
        ref = ref_data[:im.shape[0], :, i_band]
        im_corrected = im/ref
        for i_idx_sorted, idx in enumerate(idx_sorted):
            contour = contours[idx]
            mask = np.zeros_like(im_corrected)
            cv.drawContours(mask, [contour], -1, 255, -1)
            pts = np.where(mask == 255)
            roi = im_corrected[pts[0], pts[1]]
            mean_spectral = round(np.mean(roi), 4)
            if i_idx_sorted not in hyper_dict:
                hyper_dict[i_idx_sorted] = []
            hyper_dict[i_idx_sorted].append(mean_spectral)
    hyper_dict.keys()


def get_each_sample_each_band_data(data_name, data_idx_dict, ref_data, idx_sorted, contours):
    import cv2 as cv
    from tqdm import tqdm

    data = data_idx_dict[data_name]['data']
    sample_start_idx = data_idx_dict[data_name]['start_idx']

    root_dir = data_idx_dict[data_name]['data_dir']
    save_dir_origin_data = root_dir / 'origin_data'
    save_dir_origin_data.mkdir(parents=True, exist_ok=True)

    save_dir_im_data_local = root_dir / spatial_feature_type / 'im_data_local'
    save_dir_im_data_local.mkdir(parents=True, exist_ok=True)

    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]
    data_num = len(wavelengths)
    for i_band in tqdm(range(data_num), ncols=80):
        im = data[:, :, i_band]
        ref = ref_data[:im.shape[0], :, i_band]
        im_corrected = im/ref

        for i_idx_sorted, idx in enumerate(idx_sorted):
            contour = contours[idx]
            mask = np.zeros_like(im_corrected)
            cv.drawContours(mask, [contour], -1, 255, -1)
            pts = np.where(mask == 255)

            y0 = pts[0].min()
            y1 = pts[0].max()
            x0 = pts[1].min()
            x1 = pts[1].max()
            one_sample = im_corrected[y0:y1+1, x0:x1+1]

            save_dir_origin_data_one_sample = save_dir_origin_data / str(sample_start_idx + i_idx_sorted+1)
            save_dir_origin_data_one_sample.mkdir(parents=True, exist_ok=True)
            save_path = save_dir_origin_data_one_sample / f'{i_band}_{wavelengths[i_band]}.npy'
            np.save(save_path, one_sample)

            save_dir_im_data_one_sample_local = save_dir_im_data_local / str(sample_start_idx + i_idx_sorted+1)
            save_dir_im_data_one_sample_local.mkdir(parents=True, exist_ok=True)
            save_path = save_dir_im_data_one_sample_local / f'{i_band}_{wavelengths[i_band]}.png'
            one_sample = (one_sample - one_sample.min()) / (one_sample.max() - one_sample.min())
            one_sample = np.array(one_sample*255, dtype=np.uint8)
            cv.imwrite(save_path.__str__(), one_sample)


def get_all_samples_all_bands_data(show=False):
    hyper_spectral_data_dict, data_idx_dict = load_hyper_spectral_data()
    ref_data = hyper_spectral_data_dict['ref_data']
    for data_name in data_idx_dict:
        print(data_name)
        data, idx_sorted, contours = standard_pipeline_get_contours(ref_data, data_idx_dict, data_name, show=show)
        # get_mean_spectral_data(data, ref_data, idx_sorted, contours)
        get_each_sample_each_band_data(data_name, data_idx_dict, ref_data, idx_sorted, contours)


def draw_spectra_all():
    hyper_spectral_data_dict, data_idx_dict = load_hyper_spectral_data()

    data_name = 'data_bad'
    data = data_idx_dict[data_name]['data']
    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]

    from modeling_spectral_data import load_spectral_mean_data_xy
    x, y, hyper_dict_tot = load_spectral_mean_data_xy(use_part_1=1, use_part_2=1, return_dict=True)

    from modeling_spectral_data import show_hyper_dict
    type_idxes = np.arange(0, len(hyper_dict_tot))
    show_hyper_dict(x, wavelengths, title='Blueberry Spectra', type_idxes=type_idxes, alpha=0.3)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')

    # font = {'family' : 'normal',
    #        'weight' : 'normal',
    #        'size'   : 15}

    # matplotlib.rc('font', **font)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.set_xlabel('Wavelength (nm)', fontsize = 15)
    # ax.set_ylabel('Reflectance (a.u.)', fontsize = 15)
    # ax.set_title('Blueberry Spectra', fontsize = 20)
    # ax.xaxis.labelpad = 10
    # ax.yaxis.labelpad = 5
    # pl.xlabel("...", labelpad=20)
    # target_labels = ['good', 'bad']
    # targets = [0, 1]
    # colors = ['limegreen', 'salmon']
    # for target, color in zip(targets,colors):
    #    indicesToKeep = y == target
    #    #w = np.repeat(wavelengths, len(x[indicesToKeep])).reshape(-1, len(wavelengths))
    #    #ax.plot(w, x[indicesToKeep], c = color, alpha=0.5, linewidth=2, label=target_labels[target])
    #    for x_ in x[indicesToKeep]:
    #        ax.plot(wavelengths, x_, c = color, alpha=0.3, linewidth=2, label=target_labels[target])
    # ax.legend(target_labels)
    # plt.xlim([wavelengths[0], wavelengths[-1]])
    # plt.show(block=True)


def draw_spectra_mean_std():
    hyper_spectral_data_dict, data_idx_dict = load_hyper_spectral_data()

    data_name = 'data_bad'
    data = data_idx_dict[data_name]['data']
    sample_start_idx = data_idx_dict[data_name]['start_idx']
    save_name = data_idx_dict[data_name]['name']

    wavelengths = data.metadata['wavelength']
    wavelengths = [float(x) for x in wavelengths]
    data_num = len(wavelengths)

    from modeling_spectral_data import load_spectral_mean_data_xy
    x, y, hyper_dict_tot = load_spectral_mean_data_xy(use_part_1=1, use_part_2=1, return_dict=True)
    x_norm = (x-x.mean(0))/x.std(0)
    X = pd.DataFrame(x)
    X_norm = pd.DataFrame(x_norm)
    label = pd.DataFrame(y, columns=['label'])
    X.shape, y.shape

    from modeling_spectral_data import show_hyper_dict
    type_idxes = y
    from modeling_spectral_data import load_spectral_mean_data
    hyper_dict_good_tot, hyper_dict_bad_tot = load_spectral_mean_data(use_part_1=1, use_part_2=1)

    type_idxes = [1]*len(hyper_dict_bad_tot)

    good = x[y == 0]
    good_mean = good.mean(0)
    good_add_std = good_mean-good.std(0)
    good_minus_std = good_mean+good.std(0)

    bad = x[y == 1]
    bad_mean = bad.mean(0)
    bad_add_std = bad_mean-bad.std(0)
    bad_minus_std = bad_mean+bad.std(0)

    spectras = [good_mean, good_add_std, good_minus_std]+[bad_mean, bad_add_std, bad_minus_std]

    type_idxes = [0, 0, 0, 1, 1, 1]
    type_idxes = np.arange(6)
    names = ['Mean of the Sound', 'Mean-Std of the Sound', 'Mean+Std of the Sound']
    names += ['Mean of the Defected', 'Mean-Std of the Defected', 'Mean+Std of the Defected']
    names = ['Mean_Sound', 'Mean_minus_Std_Sound', 'Mean_add_Std_Sound']
    names += ['Mean_Defected', 'Mean_minus_Std_Defected', 'Mean_add_Std_Defected']
    from scipy.io import savemat
    mean_std_data = {'wavelengths': wavelengths, names[0]: good_mean, names[1]: good_add_std,names[2]: good_minus_std,names[3]: bad_mean,names[4]: bad_add_std,names[5]: bad_minus_std,}
    # savemat(r'D:\test\mean_std_data.mat', mean_std_data)
    show_hyper_dict_mean_std(spectras, wavelengths, title='420 Blueberries Spectra', type_idxes=type_idxes, alpha=0.3, names=names)


def main():
    #get_all_samples_all_bands_data(show=False)
    #draw_spectra_all()
    draw_spectra_mean_std()
    pass


if __name__ == '__main__':
    main()
