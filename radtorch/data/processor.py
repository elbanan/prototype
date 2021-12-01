from ..core.const import *
from ..core.utils import *


class DICOMProcessor(Dataset):
    def __init__(
        self,
        root,
        ext="dcm",
        type="file",
        table=None,
        class_to_idx=None,
        path_col=None,
        label_col=None,
        study_col=None,
        transform=None,
        num_output_channels=1,
        window=None,
        level=None,
        split=None,
    ):
        self.ext = ext
        self.type = type
        if root.endswith("/"):
            self.root = root
        else:
            self.root = root + "/"
        self.num_output_channels = num_output_channels
        self.window = window
        self.level = level
        self.class_to_idx = class_to_idx

        if path_col:
            self.path_col = path_col
        else:
            self.path_col = "path"

        if label_col:
            self.label_col = label_col
        else:
            self.label_col = "label"

        if path_col:
            self.study_col = study_col
        else:
            self.study_col = "study_id"

        if isinstance(table, pd.DataFrame):
            self.table = table
            if self.class_to_idx:
                self.classes = [k for k, v in self.class_to_idx.items()]
            else:
                self.classes = self.table[self.label_col].unique().tolist()
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
        else:
            if self.type == "file":
                if self.class_to_idx:
                    self.classes = [k for k, v in self.class_to_idx.items()]
                else:
                    self.classes, self.class_to_idx = find_classes(self.root)
                self.table = root_to_data(
                    root=self.root,
                    ext=self.ext,
                    path_col=self.path_col,
                    label_col=self.label_col,
                )
                if len(self.table) == 0:
                    raise ValueError(
                        "No .{:} files were found in {:}. Please check.".format(
                            ext, self.root
                        )
                    )
            elif self.type == "directory":
                self.classes, self.class_to_idx, self.table = self.dicom_dir_to_table(
                    self.root, self.ext
                )

        if transform == None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = transform

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

    def info(self):
        info = pd.DataFrame.from_dict(
            ({key: str(value) for key, value in self.__dict__.items()}).items()
        )
        info.columns = ["Property", "Value"]
        return info

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        path = self.table.iloc[idx][self.path_col]
        label = self.table.iloc[idx][self.label_col]
        label_id = self.class_to_idx[label]

        if self.type == "file":
            if self.ext != "dcm":
                img = Image.open(P)
            else:
                img = dicom_handler(
                    img_path=P,
                    num_output_channels=self.num_output_channels,
                    WW=self.window,
                    WL=self.level,
                )
            if self.transforms:
                img = self.transforms(img)

        elif self.type == "directory":
            img_list = sorted(
                [
                    (file, float(pydicom.read_file(file).SliceLocation))
                    for file in glob.glob(
                        path + "/" + "**/*." + self.ext, recursive=True
                    )
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            img_list = [i[0] for i in img_list]
            img = [
                dicom_handler(
                    img_path=i,
                    num_output_channels=self.num_output_channels,
                    WW=self.window,
                    WL=self.level,
                )
                for i in img_list
            ]
            if self.transforms:
                img = [self.transforms(i) for i in img]
                img = torch.stack(img)

        try:
            uid = self.table.iloc[idx]["uid"]
        except:
            uid = self.table.index.values.tolist()[idx]
        return uid, img, label_id

    def get_loaders(self, batch_size=16, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)


# Allow to select subset of the CT stack
# check 3 w/l on  CT stack
