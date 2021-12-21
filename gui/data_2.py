import streamlit as st
import torch, torchvision, time


def app():
    st.header('Create New Dataset +2')

    st.write("  ")
    st.write("  ")
    st.subheader('General Attributes')
    root= st.text_input(label='Folder', placeholder='directory/folder containing your data files.', help='Directory/folder containing your data files.')
    col1, col2= st.columns(2)
    with col1: name= st.text_input(label='Name', placeholder='Name given to your dataset.', help='Nickname of dataset.')
    with col2: type=st.selectbox(label='Dataset Type', options=['files','directory'], help='Type of data instance. Either DICOM single images or DICOM directory.')
    label_table=st.file_uploader(label='Label Table (optional)', help='CSV file containing information about the training data including file path and label for images.')
    col3, col4, col5 = st.columns(3)
    with col3: path_col=st.text_input(label='Image Path Column', value='path', help='Name of the column containing iamge path in your label table.')
    with col4: label_col=st.text_input(label='Image Label Column', value='label', help='Name of the column containing labels in your label table.')
    with col5: study_col=st.text_input(label='Study ID Column', value='study_id', help='Name of the column containing ids for DICOM studies in your label table.')


    st.write("  ")
    st.write("  ")
    st.subheader('DICOM Attributes')
    col6, col7, col8 = st.columns(3)
    with col6:
        num_output_channels=st.selectbox(label='Number of Output Image Channels', options=[1, 2, 3], help='Number of expected image channels after transformations.')

    with col7:
        window_level=st.selectbox(label='Window/Level', options=[False, True], index=0, help='Window/Level DICOM images/channels. See https://radiopaedia.org/articles/windowing-ct?lang=us')

    with col8:
        st.empty()
        st.write("  ")
        st.write("  ")
        if window_level:
            with st.expander("Window/Level Channel 1"):
                WW1 = st.number_input(label='Window Width (Channel 1)', step=1, help='Window width for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WL1 = st.number_input(label='Window Level (Channel 1)', step=1, help='Window level for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
            if num_output_channels in [2,3]:
                with st.expander("Window/Level Channel 2"):
                    WW2 = st.number_input(label='Window Width (Channel 2)', step=1, help='Window width for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                    WL2 = st.number_input(label='Window Level (Channel 2)', step=1, help='Window level for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
            if num_output_channels == 3:
                with st.expander("Window/Level Channel 3"):
                    WW3 = st.number_input(label='Window Width (Channel 3)', step=1, help='Window width for DICOM image - Channel #3. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                    WL3 = st.number_input(label='Window Level (Channel 3)', step=1, help='Window level for DICOM image - Channel #3. See https://radiopaedia.org/articles/windowing-ct?lang=us')


    show_wl = st.checkbox(label='Display Different Window/Level', value=False)
    if show_wl:
        ct_wl = {"brain": {"window":80, "level":40},
        "subdural": {"window":200, "level":75},
        "stroke_1": {"window":8, "level":32},
        "stroke_2": {"window":40, "level":40},
        "temporal_1": {"window":2800, "level":600},
        "temporal_2": {"window":4000, "level":700},
        "head_soft": {"window":380, "level":40},
        "lungs": {"window":1500, "level":-600},
        "mediastinum": {"window":350, "level":50},
        "abdomen_soft": {"window":350, "level":50},
        "liver": {"window":150, "level":30},
        "spine_soft": {"window":250, "level":50},
        "spine_bone": {"window":1800, "level":400},}
        t = st.dataframe(ct_wl)


    st.write("  ")
    st.write("  ")
    st.subheader('Transformations/Augmentations')
    cola, colb, colc = st.columns(3)
    with cola: t = st.selectbox(label='Use Transformations ?', options=[False, True], index=0)
    with colb: st.empty()
    with colc: st.empty()
    if t:
        with st.spinner('Loading transformation modules'):
            time.sleep(0.5)
        col9, col10, col11 = st.columns(3)
        with col9:
            if t:
                rs = st.selectbox(label='Image Resize', options=[False, True], index=0, help='Resize images to a new custom size.')
                with st.expander("options"):
                    rs_h = st.number_input(label='Image Resize (Height)', step=1, help='New image height')
                    rs_w = st.number_input(label='Image Resize (Width)', step=1, help='New image width')
                st.empty()
                rc = st.selectbox(label='Random Crop', options=[False, True], index=0, help='Randomly Crop the given image at a random location. See https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop')
                with st.expander("options"):
                    rc_h = st.number_input(label='Crop height', step=1, help='Crop height')
                    rc_w = st.number_input(label='Crop width', step=1, help='Crop width')

        with col10:
            n = st.selectbox(label='Normalize', options=[False, True], index=0, help='Normalize your dataset to a new mean and standard deviation. Please note that images with new window/level are automatically normalized between 0-1.')
            with st.expander("options"):
                n_mean = st.number_input(label='Normalize Mean', step=0.1, help='New image mean.')
                n_std = st.number_input(label='Normalize Standard Deviation', step=0.1, help='New image standard deviation.')
            rhf = st.selectbox(label='Random Horizontal Flip', options=[False, True], index=0, help='Randomly Horizontally flip the given image randomly with a given probability. See https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip')
            with st.expander("options"):
                rhf_p = st.number_input(label='Horizontal Flip Probability', step=0.1, max_value=1.0, help='Probability')

        with col11:
            rr = st.selectbox(label='Random Rotation', options=[False, True], index=0, help='Randomly Rotate the image by angle. See https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomRotation')
            with st.expander("options"):
                pos_rr = st.number_input(label='Rotation +degrees', step=1, help='+degrees')
                neg_rr = st.number_input(label='Rotation -degrees', step=1, help='-degrees')
            rvf = st.selectbox(label='Random Vertical Flip', options=[False, True], index=0, help='Randomly Vertically flip the given image randomly with a given probability. See https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip')
            with st.expander("options"):
                rvf_p = st.number_input(label='Vertical Flip Probability', step=0.1, max_value=1.0, help='Probability')


    st.write("  ")
    st.write("  ")
    st.subheader('Data Management')
    col12, col13, col14 = st.columns(3)
    with col12:
        sample = st.number_input(label='Sample from Dataset', min_value=0.1, value=1.0, step=0.1, max_value=1.0, help='Percentage of your dataset to be used.')
        batch_size = st.number_input(label='Batch Size', min_value=1, value=16, step=1, help='Percentage of your dataset to be used.')
    with col13:
        valid = st.number_input(label='Validation Percent', step=0.1, min_value=0.0,  value=0.2, max_value=1.0, help='Percentage of your dataset to be used for Validation.')
        train_balance=st.selectbox(label='Balance Train Subset', index=0, options=[False, 'upsample', 'downsample'], help='In case of imbalanced datasets where training classes have unequal number of images/studies, this gives the option to equalize the number of images/studies in "train" subset only. See https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/')
    with col14:
        test = st.number_input(label='Test Percent', step=0.1, value=0.2, min_value=0.0, max_value=1.0, help='Percentage of your dataset to be used for Testing.')
        ignore_zero_img= st.selectbox(label='Ignore Zero/Empty Images', options=[False, True], index=0, help='Select this to ignore empty images from your dataset.')
        st.write("  ")
        st.write("  ")
        st.write("  ")
        st.write("  ")
        submitted = st.button("Create Dataset")

    if submitted:

        if window_level:
            if num_output_channels==1:
                window = WW1
                level = WL1
            elif num_output_channels==2:
                window = [WW1, WW2]
                level = [WL1, WL2]
            elif num_output_channels==3:
                window = [WW1, WW2, WW3]
                level = [WL1, WL2, WL3]
        else:
            window=None
            level=None

        if t:
            transforms_list=[]
            if rs: transforms_list.append(torchvision.transforms.Resize(size=(rs_h, rs_w)))
            if rc: transforms_list.append(torchvision.transforms.RandomCrop(size=(rc_h, rc_w)))
            if rhf: transforms_list.append(torchvision.transforms.RandomHorizontalFlip(p=rhf_p))
            if rvf: transforms_list.append(torchvision.transforms.RandomVerticalFlip(p=rvf_p))
            if n: transforms_list.append(torchvision.transforms.Normalize(n_mean, n_std))
            transforms = torchvision.transforms.Compose(transforms_list)
            transforms = {'train':transforms, 'valid':transforms, 'test':transforms}
        else:
            transforms = False


        if sample == 1.0:sample=False

        data_info = {
        'name':name,
        'root':root,
        'type':type,
        'label_table':label_table,
        'path_col':path_col,
        'label_col':label_col,
        'study_col':study_col,
        'num_output_channels':num_output_channels,
        'WW':window,
        'WL': level,
        'transform':transforms,
        'split':{'valid':valid, 'test':test},
        'train_balance':train_balance,
        'batch_size':batch_size,
        'ignore_zero_img':ignore_zero_img,
        'sample':sample
        }

        st.write(data_info)
