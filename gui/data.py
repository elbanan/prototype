import streamlit as st



def app():
    with st.form("create_dataset_form"):
        st.header('Create New Dataset +')

        st.write("  ")
        st.write("  ")
        st.subheader('General Attributes')
        col1, col2= st.columns(2)
        with col1: root= st.text_input(label='Folder', placeholder='directory/folder containing your data files.', help='Directory/folder containing your data files.')
        with col2: type=st.selectbox(label='Dataset Type', options=['DICOM files','DICOM Directory'], help='Type of data instance. Either DICOM single images or DICOM directory.')
        label_table=st.file_uploader(label='Label Table (optional)', help='CSV file containing information about the training data including file path and label for images.')
        col3, col4, col5 = st.columns(3)
        with col3: path_col=st.text_input(label='Image Path Column', value='path', help='Name of the column containing iamge path in your label table.')
        with col4: label_col=st.text_input(label='Image Label Column', value='label', help='Name of the column containing labels in your label table.')
        with col5: study_col=st.text_input(label='Study ID Column', value='study_id', help='Name of the column containing ids for DICOM studies in your label table.')



        st.write("  ")
        st.write("  ")
        st.subheader('DICOM Attributes')
        col6, col7, col8 = st.columns(3)
        with col6: num_output_channels=st.selectbox(label='Number of Output Image Channels', options=[1, 2, 3], help='Number of expected image channels after transformations.')


        if num_output_channels==1:
            with col7: WW1 = st.number_input(label='Window Width', step=1, help='Window width for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
            with col8: WL1 = st.number_input(label='Window Level', step=1, help='Window level for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
        elif num_output_channels==2:
            with col7:
                WW1 = st.number_input(label='Window Width (Channel 1)', step=1, help='Window width for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WW2 = st.number_input(label='Window Width (Channel 2)', step=1, help='Window width for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
            with col8:
                WL1 = st.number_input(label='Window Level (Channel 1)', step=1, help='Window level for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WL2 = st.number_input(label='Window Level (Channel 2)', step=1, help='Window level for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
        elif num_output_channels==3:
            with col7:
                WW1 = st.number_input(label='Window Width (Channel 1)', step=1, help='Window width for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WW2 = st.number_input(label='Window Width (Channel 2)', step=1, help='Window width for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WW3 = st.number_input(label='Window Width (Channel 3)', step=1, help='Window width for DICOM image - Channel #3. See https://radiopaedia.org/articles/windowing-ct?lang=us')
            with col8:
                WL1 = st.number_input(label='Window Level (Channel 1)', step=1, help='Window level for DICOM image - Channel #1. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WL2 = st.number_input(label='Window Level (Channel 2)', step=1, help='Window level for DICOM image - Channel #2. See https://radiopaedia.org/articles/windowing-ct?lang=us')
                WL3 = st.number_input(label='Window Level (Channel 3)', step=1, help='Window level for DICOM image - Channel #3. See https://radiopaedia.org/articles/windowing-ct?lang=us')




        st.write("  ")
        st.write("  ")
        st.subheader('Transformations/Augmentations')

        col9, col10, col11 = st.columns(3)
        with col9:
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
            sample = st.number_input(label='Sample from Dataset', step=0.1, max_value=1.0, help='Percentage of your dataset to be used.')
            batch_size = st.number_input(label='Batch Size', min_value=1, step=1, help='Percentage of your dataset to be used.')
        with col13:
            valid = st.number_input(label='Validation Percent', step=0.1, max_value=1.0, help='Percentage of your dataset to be used for Validation.')
            train_balance=st.selectbox(label='Balance Train Subset', index=0, options=[False, 'upsample', 'downsample'], help='In case of imbalanced datasets where training classes have unequal number of images/studies, this gives the option to equalize the number of images/studies in "train" subset only. See https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/')
        with col14:
            test = st.number_input(label='Test Percent', step=0.1, max_value=1.0, help='Percentage of your dataset to be used for Testing.')
            ignore_zero_img= st.selectbox(label='Ignore Zero/Empty Images', options=[False, True], index=0, help='Select this to ignore empty images from your dataset.')
        st.write("  ")
        st.write("  ")
        submitted = st.form_submit_button("Create Dataset")
        if submitted:
            if num_output_channels==1:
                window = WW1
                level = WL1
            elif num_output_channels==2:
                window = [WW1, WW2]
                level = [WL1, WL2]
            elif num_output_channels==3:
                window = [WW1, WW2, WW3]
                level = [WL1, WL2, WL3]

            data_info = {
            'name':'dataset1',
            'root':root,
            'type':type,
            'label_table':label_table,
            'path_col':path_col,
            'label_col':label_col,
            'study_col':study_col,
            'num_output_channels':num_output_channels,
            'WW':window,
            'WL': level,
            # ext:,

            # ext="dcm",type="file",label_table=None,path_col="path",study_col="study_id",label_col="label",num_output_channels=1,transform=False,WW=None,WL=None,split=None,ignore_zero_img=False,sample=False,train_balance=False,batch_size=16,output_subset="all"):

            }
            st.write(data_info)
