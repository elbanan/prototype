from ..core.const import *
from ..core.utils import *
from .processor import *


class DICOMDataset:
    """
    DICOM Dataset class. This class is a container of all methods and functionalities needed to create `dataloaders` that will be used in the trainin process.
    Its core is the `DICOMprocessor` which is an extension of pytorch `Dataset` class and utilizes pytorch with custom functions from `core` to handle dicom files.

    Parameters
    ----------

    root: str
        root/directory containing image files. If no label table is specified, classes and paths are extracted automatically as follows:

        For `type` of "files", the expected directory structure should be:
        Parent folder / class_1 / images
                      / class_2 / images
                      ....

        For `type`of "directory", the expected directory structure should be:
        Parent folder / class_1 / study_1 / images
                      / class_1 / study_2 / images
                      / class_2 / study_3 / images
                      / class_2 / study_4 / images
                      ....

    ext: str
        type/file extension of images. default='dcm'

    type: str
        type of dataset instances. either images 'files' or 'directory' as outlined above. default='file'.

    label_table: (optional)

    path_col: str
        name of the table column containing the image path. default='path'.

    label_col: str
        name of the table column containing the image label. default='label'.

    study_col: str
        name of the table column containing the study id. default='study_id'.

    transform: dictionary of pytorch transforms (optional)
        dictionary of lists of pytorch transforms to be applied to images. default='False' which automatically transforms all images into tensors without extra transformations.
        Expected dictionary follows the following structure:
        transform = {
            'train': transforms.Compose([.....]),
            'valid': transforms.Compose([.....]),
            'test': transforms.Compose([....]),
        }
        with 'valid' and 'test' being optional if dataset not splitted.

        See https://pytorch.org/vision/stable/transforms.html

    num_output_channels: int (optional)
        number of expected image channels after transforms. default=1.
        Notice: by default, most greyscale DICOM images have 1 channel and most pre-trained pytorch neural network models expect 3 channel input.

    WW: int or list of int (optional)
        value for DICOM window width. Number of integers must be equal to number of channels expected. default=None
        See https://radiopaedia.org/articles/windowing-ct?lang=us

    WL: int or list of int (optional)
        value for DICOM window level. Number of integers must be equal to number of channels expected. default=None
        See https://radiopaedia.org/articles/windowing-ct?lang=us

    split: dict (optional)
        dictionary of 'valid' and 'test' decimal/float to be used to split the data for training/testing. Accepts both 'valid' and 'test' or 'valid' only. default=None
        e.g. split = {'valid': 0.2, 'test': 0.1}

    ignore_zero_img: boolean (Optional)
        option to ignore images containing all empty or 0-value pixel. default=False.

    sample: int (optional)
        option to use sample of the whole dataset. Useful for quick algorith testing with small portion of the main dataset. default=False.

    train_balance: str (optional)
        In case of imbalanced datasets where training classes have unequal number of images/studies, this gives the option to equalize the number of images/studies in `training` subset only.
        This can be set to `upsample` or `downsample`. default=False.
        See https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

    batch_size: int
        batch size for the generated dataloaders. default=16.

    output_subset: str
        dataloader to be generated. default='all' which generates all 'train', 'valid' and 'test'


    Attributes
    ----------
    data_table: dict
        dictionary of all generated data tables of 'train' and 'valid' and 'test' if exist.
        Follows the sturcture: DICOMDataset.data_table = {'train': pandas dataframe, 'valid': pandas dataframe, 'test': pandas dataframe}

    classes: list
        list of generated classes/labels.

    class_to_idx: dict
        dictionary of generated classes/labels and their assigned numerical ids.

    loaders: dict
        dictionary containing the generated dataloaders following the structure: {'train': dataloader, 'valid': dataloader, 'test': dataloader}

    """

    def __init__(self,root,ext="dcm",type="file",label_table=None,path_col="path",study_col="study_id",label_col="label",num_output_channels=1,transform=False,WW=None,WL=None,split=None,ignore_zero_img=False,sample=False,train_balance=False,batch_size=16,output_subset="all"):

        if root.endswith("/"):
            self.root = root
        else:
            self.root = root + "/"

        self.ext = ext
        self.type = type
        self.label_col = label_col
        self.path_col = path_col
        self.study_col = study_col
        self.num_output_channels = num_output_channels

        self.idx = {}
        self.ignore_zero_img = ignore_zero_img
        self.sample = sample
        self.train_balance = train_balance
        self.batch_size = batch_size

        self.window, self.level = check_wl(WW, WL)

        if isinstance(label_table, dict):  # NEEDS CHECK FOR DICOM DIRECTORY STRUCTURE
            self.data_table, self.classes, self.class_to_idx = dict_to_data(
                table_dict=label_table, classes=self.classes, label_col=self.label_col
            )
            for k, v in self.data_table:
                v = add_uid_column(v, length=10)

        else:
            if isinstance(label_table, pd.DataFrame):
                self.data_table = {}
                table = label_table
                self.classes = label_table[self.label_col].unique().tolist()
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
            else:
                self.data_table = {}
                if self.type == "file":
                    self.classes, self.class_to_idx = find_classes(self.root)
                    table = root_to_data(
                        root=self.root,
                        ext=self.ext,
                        path_col=self.path_col,
                        label_col=self.label_col,
                    )
                    if len(table) == 0:
                        raise ValueError(
                            "No .{:} files were found in {:}. Please check.".format(
                                self.ext, self.root
                            )
                        )
                elif self.type == "directory":
                    self.classes, self.class_to_idx, table = self.dicom_dir_to_table(
                        self.root, self.ext
                    )

            if split:
                self.valid_percent = split["valid"]
                if "test" in split:
                    self.test_percent = split["test"]
                    self.train_percent = 1.0 - (self.valid_percent + self.test_percent)
                    self.idx["test"], self.idx["valid"], self.idx["train"] = split_data(
                        table,
                        valid_percent=self.valid_percent,
                        test_percent=self.test_percent,
                    )
                    for i in subsets:
                        self.data_table[i] = table.loc[self.idx[i], :]
                        self.data_table[i] = add_uid_column(
                            self.data_table[i], length=10
                        )
                else:
                    self.train_percent = 1.0 - self.valid_percent
                    self.idx["train"], self.idx["valid"] = split_data(
                        table, valid_percent=self.valid_percent, test_percent=False
                    )
                    for i in subsets[:2]:
                        self.data_table[i] = table.loc[self.idx[i], :]
                        self.data_table[i] = add_uid_column(
                            self.data_table[i], length=10
                        )

            else:
                self.data_table["train"] = table

        if self.ignore_zero_img:
            if self.type != "file":
                raise ValueError(
                    'Ignore empty images feature is currently available with DICOMDatasets with "file" type.'
                )
            else:
                for i in self.ignore_zero_img:
                    self.data_table[i] = check_zero_image(
                        table=self.data_table[i], path_col=self.path_col
                    )

        if self.sample:
            for k, v in self.data_table.items():
                self.data_table[k] = v.sample(frac=self.sample, random_state=100)

        if self.train_balance:
            self.data_table["train"] = balance(
                df=self.data_table["train"],
                method=self.train_balance,
                label_col=self.label_col,
                classes=self.classes,
            )

        self.data_table["train"] = add_uid_column(self.data_table["train"], length=10)

        if isinstance(transform, dict):
            self.transforms = transform
        else:
            self.transforms = {
                k: transforms.Compose([transforms.ToTensor()])
                for k, v in self.data_table.items()
            }

        self.loaders = self.get_loaders(batch_size=batch_size, subset=output_subset)
        if self.type == "file":
            self.img_size = self.loaders["train"].dataset[0][1].shape[1]

    def info(self):
        """
        Returns class relevant information/parameters
        """

        info = pd.DataFrame.from_dict(
            ({key: str(value) for key, value in self.__dict__.items()}).items()
        )
        info.columns = ["Property", "Value"]
        for i in ["train", "valid", "test"]:
            try:
                info.loc[len(info.index)] = [
                    i + " dataset size",
                    len(self.data_table[i]),
                ]
            except:
                pass
        return info

    def dicom_dir_to_table(self, root, ext):
        classes, class_to_idx = find_classes(root)
        table = pd.DataFrame(
            columns=[self.study_col, self.path_col, "num_images", self.label_col]
        )
        for c in classes:
            study_dir = os.path.join(root, c)
            study_idx = [x for x in os.walk(os.path.join(root, c))][0][1]
            study_paths = [x[0] for x in os.walk(os.path.join(root, c))][1:]
            for i in range(len(study_idx)):
                table.loc[len(table.index)] = [
                    study_idx[i],
                    study_paths[i],
                    len(
                        [
                            file
                            for file in glob.glob(
                                study_paths[i] + "/" + "**/*." + ext, recursive=True
                            )
                        ]
                    ),
                    c,
                ]
        return classes, class_to_idx, table

    def get_loaders(self, batch_size=16, shuffle=True, subset="all"):
        """
        Generates dataloaders from the DICOMDataset object.

        Parameters
        ----------
        batch_size: int
            batch size for the generated dataloaders. default=16.

        shuffle: boolean
            Shuffle images each with each use of dataloader.

        subset: str
            dataloader to be generated. default='all' which generates all 'train', 'valid' and 'test'.

        Returns
        -------
        Dictionary of dataloaders from DICOMDataset object as :  {'train': dataloader, 'valid': dataloader, 'test': dataloader}
        """

        if subset == "all":
            output = {}
            for k, v in self.data_table.items():
                output[k] = DICOMProcessor(
                    root=self.root,
                    ext=self.ext,
                    num_output_channels=self.num_output_channels,
                    table=v,
                    type=self.type,
                    class_to_idx=self.class_to_idx,
                    path_col=self.path_col,
                    study_col=self.study_col,
                    label_col=self.label_col,
                    transform=self.transforms[k],
                    window=self.window,
                    level=self.level,
                ).get_loaders(batch_size=batch_size, shuffle=shuffle)
            return output
        else:
            return {
                subset: DICOMProcessor(
                    root=self.root,
                    ext=self.ext,
                    num_output_channels=self.num_output_channels,
                    table=self.data_table[subset],
                    type=self.type,
                    class_to_idx=self.class_to_idx,
                    study_col=self.study_col,
                    path_col=self.path_col,
                    label_col=self.label_col,
                    transform=self.transforms[subset],
                    window=self.window,
                    level=self.level,
                ).get_loaders(batch_size=batch_size, shuffle=shuffle)
            }

    def view_batch(self,data="train",figsize=(25, 5),study_id=0,rows=2,batch_size=16,shuffle=True,num_images=False,cmap="gray"):

        """
        Displays a batch or sample of batch of the DICOM dataset.

        Parameters
        ----------

        data: str
            dataloader subset to be displayed. default='train'

        figsize: tuple
            size of the output matplot figure. default=(25,5)

        study_id: int
            id of the study to be displayed. Only with DICOMDataset type = 'directory', otherwise ignored.

        rows: int
            number of rows. default=2.

        batch_size: int
            batch size, default=16.

        shuffle: int
            Shuffle images each with each use of dataloader.

        num_images: int
            number of images to be displayed. Useful when batch size is big and you want to display only a sample of a batch. default=False

        cmap: str
            color map for the generated figure. default = 'gray'. Please refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html

        """

        loader = DICOMProcessor(
            root=self.root,
            ext=self.ext,
            num_output_channels=self.num_output_channels,
            table=self.data_table[data],
            type=self.type,
            class_to_idx=self.class_to_idx,
            path_col=self.path_col,
            study_col=self.study_col,
            label_col=self.label_col,
            transform=self.transforms[data],
            window=self.window,
            level=self.level,
        ).get_loaders(batch_size=batch_size, shuffle=shuffle)

        if self.type == "file":
            uidx, images, labels = (iter(loader)).next()
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            batch = images.shape[0]
            if num_images:
                if num_images > batch:
                    print(
                        "Warning: Selected number of images is less than batch size. Displaying a batch instead."
                    )
                else:
                    batch = num_images

            fig = plt.figure(figsize=figsize)
            for i in np.arange(batch):
                ax = fig.add_subplot(
                    rows, int(batch / rows), i + 1, xticks=[], yticks=[]
                )
                if images[i].shape[0] == 3:
                    ax.imshow(np.transpose(images[i], (1, 2, 0)), cmap=cmap)
                elif images[i].shape[0] == 1:
                    ax.imshow(np.squeeze(images[i]), cmap=cmap)
                ax.set_title(self.classes[labels[i]])
        elif self.type == "directory":
            imgs = loader.dataset[study_id][1]
            if num_images:
                imgs = imgs[:num_images]
            grid = make_grid(imgs)
            show_stack(grid, figsize)

    def view_multichannel_image(self,data="train",idx=0,figsize=(25, 5),batch_size=16,shuffle=True,cmap="gray"):
        """
        Displays different channels of a multi-channel image. Supports up to 3 channels for now.

        Parameters
        ----------

        data: str
            dataloader subset to be displayed. default='train'.

        idx: int
            index of the image to be displayed.

        figsize: tuple
            size of the output matplot figure. default=(25,5).

        batch_size: int
            batch size, default=16.

        shuffle: int
            Shuffle images each with each use of dataloader.

        cmap: str
            color map for the generated figure. default = 'gray'. Please refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html

        """

        loader = DICOMProcessor(
            root=self.root,
            ext=self.ext,
            table=self.data_table[data],
            type="file",
            class_to_idx=self.class_to_idx,
            path_col=self.path_col,
            label_col=self.label_col,
            transform=self.transforms[data],
            num_output_channels=self.num_output_channels,
            window=self.window,
            level=self.level,
        ).get_loaders(batch_size=batch_size, shuffle=shuffle)
        uidx, images, labels = (iter(loader)).next()

        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        img = images[idx]

        if img.shape[0] == 1:
            print("Warning: Selected image does not have 3 channels. Please check.")

        num_channels = img.shape[0]
        fig = plt.figure(figsize=figsize)
        img = np.transpose(img, (1, 2, 0))

        channels = [img[:, :, i] for i in range(0, num_channels)]

        for i in range(0, num_channels):
            ax = fig.add_subplot(1, num_channels, i + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(channels[i]), cmap=cmap)
            ax.set_title("channel " + str(i))

    def header_info(self, data="train", limit=10):
        """
        Returns a table/pandas dataframe of DICOM header for all images in dataset/subset.

        Parameters
        ----------

        data: str
            dataloader subset to be displayed. default='train'

        limit: int
            limit of images/instances to include. default=10

        Returns
        -------
        table/pandas dataframe of DICOM header for all images in dataset/subset.
        """

        table = self.data_table[data]
        header_col = []
        for c in self.classes:
            g = pydicom.read_file(
                (table[table[self.label_col] == c]).iloc[0][self.path_col]
            )
            header_col += [g[k].keyword for k in g.keys()]
        df = pd.DataFrame(columns=(list(set(header_col))).sort())
        s = 0
        for i, r in self.data_table[data].iterrows():
            if s in range(0, limit):
                d = pydicom.read_file(r[self.path_col])
                df = df.append(
                    dict(
                        [
                            (d[k].keyword, d[k].value)
                            for k in d.keys()
                            if d[k].keyword != "PixelData"
                        ]
                    ),
                    ignore_index=True,
                )
                s += 1
        return df

    def examine_img(self,data="train",figure_size=(15, 15),resize=(128, 128),img_idx=0,cmap="gray"):
        img = pydicom.read_file(self.data_table[data][self.path_col].tolist()[img_idx]).pixel_array
        img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_CUBIC)
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(
                    str(val),
                    xy=(y, x),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if img[x][y] < thresh else "black",
                )

    def data_stat(self, plot=False, figure_size=(8, 6)):
        """
        Displays different class distribution per subset.

        Parameters
        ----------
        plot: boolean
            True to display data statistics as figure. default=False

        figure_size: tuple
            size of the displayed figure. default=(8, 6)


        Returns
        -------
        Padas dataframe if plot is set to False.
        """

        d, c, i, n = [], [], [], []
        for k, v in self.data_table.items():
            for l, j in self.class_to_idx.items():
                d.append(k)
                c.append(l)
                i.append(j)
                n.append(v[self.label_col].value_counts()[l].sum())
        df = pd.DataFrame(
            list(zip(d, c, i, n)), columns=["Dataset", "Class", "Class_idx", "Count"]
        )
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figure_size)
            ax = sns.barplot(
                x="Dataset", y="Count", hue="Class", data=df, palette="viridis"
            )
            show_values_on_bars(ax)
        else:
            return df
