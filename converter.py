import xml.etree.ElementTree as ET
from pathlib import Path, PurePath
from typing import List
import os, sys, glob
import pandas as pd
import numpy as np

from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import list_files_recursively, load_json, save_json
import json
from tqdm import tqdm
import yaml
import shutil
import csv

from dataset import Dataset

class labelme2coco:
    def __init__(self):
        print("ready to convert labelme data 2 coco")

    def get_coco_from_labelme_folder(
        self,
        labelme_folder: str,
        coco_category_list: List = None
    ) -> Coco:
        """
        Args:
            labelme_folder: folder that contains labelme annotations and image files
            coco_category_list: start from a predefined coco cateory list
        """
        # get json list
        _, abs_json_path_list = list_files_recursively(
            labelme_folder, contains=[".json"]
        )
        labelme_json_list = abs_json_path_list
        labelme_json_list.sort()

        # init coco object
        coco = Coco()

        if coco_category_list is not None:
            coco.add_categories_from_coco_category_list(coco_category_list)

        # parse labelme annotations
        category_ind = 0
        for json_path in tqdm(
            labelme_json_list, "Converting labelme annotations to COCO format"
        ):
            try:
                data = load_json(json_path)
                
                # get image size
                image_path = str(Path(labelme_folder) / data["imagePath"])
                
                # use the image sizes provided by labelme (they already account for
                # things such as EXIF orientation)
                width = data["imageWidth"]
                height = data["imageHeight"]
                # init coco image
                coco_image = CocoImage(
                    file_name=data["imagePath"], height=height, width=width
                )
                # iterate over annotations
                for shape in data["shapes"]:
                    # set category name and id
                    category_name = shape["label"]
                    
                    category_id = None
                    for (
                        coco_category_id,
                        coco_category_name,
                    ) in coco.category_mapping.items():
                        if category_name == coco_category_name:
                            category_id = coco_category_id
                            break
                    # add category if not present
                    if category_id is None:
                        category_id = category_ind
                        coco.add_category(
                            CocoCategory(id=category_id, name=category_name)
                        )
                        category_ind += 1
    
                    # circles and lines to segmentation
                    if shape["shape_type"] == "circle":
                        (cx, cy), (x1, y1) = shape["points"]
                        r = np.linalg.norm(np.array([x1 - cx, y1 - cy]))
                        angles = np.linspace(0, 2 * np.pi, 50 * (int(r) + 1))
                        x = cx + r * np.cos(angles)
                        y = cy + r * np.sin(angles)
                        points = np.rint(np.append(x, y).reshape(-1, 2, order="F"))
                        _, index = np.unique(points, return_index=True, axis=0)
                        shape["points"] = points[np.sort(index)]
                        shape["shape_type"] = "polygon"
                    elif shape["shape_type"] == "line":
                        (x1, y1), (x2, y2) = shape["points"]
                        shape["points"] = [
                            x1,
                            y1,
                            x2,
                            y2,
                            x2 + 1e-3,
                            y2 + 1e-3,
                            x1 + 1e-3,
                            y1 + 1e-3,
                        ]
                        shape["shape_type"] = "polygon"
    
                    # parse bbox/segmentation
                    if shape["shape_type"] == "rectangle":
                        x1 = shape["points"][0][0]
                        y1 = shape["points"][0][1]
                        x2 = shape["points"][1][0]
                        y2 = shape["points"][1][1]
                        coco_annotation = CocoAnnotation(
                            bbox=[x1, y1, x2 - x1, y2 - y1],
                            category_id=category_id,
                            category_name=category_name,
                        )
                    elif shape["shape_type"] == "polygon":
                        segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                        coco_annotation = CocoAnnotation(
                            segmentation=segmentation,
                            category_id=category_id,
                            category_name=category_name,
                        )
                    else:
                        raise NotImplementedError(
                            f'shape_type={shape["shape_type"]} not supported.'
                        )
                    coco_image.add_annotation(coco_annotation)
                coco.add_image(coco_image)
            except Exception as e:
                continue
        
        return coco

    def convert(
        self,
        input_dir:str,
        output_dir: str,
        train_split_rate: float = 1
    ):        
        """
        Args:
            input_dir: folder that contains labelme annotations and image files
            output_dir: path for coco jsons to be exported
            train_split_rate: ration fo train split
        """
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        json_extensions = [".json"]
        add_path = None
        image_dir = None

        for dirpath, dirnames, filenames in os.walk(input_dir):
            if any(filename.endswith('.json') for filename in filenames) and any(filename.lower().endswith(('.png', '.jpg', '.jpeg')) for filename in filenames):
                input_dir = dirpath
                break
                
        coco = self.get_coco_from_labelme_folder(labelme_folder=input_dir)

        if 0 < train_split_rate < 1:
            result = coco.split_coco_as_train_val(train_split_rate)
            # export train split
            save_path = str(Path(output_dir) / "train.json")
            save_json(result["train_coco"].json, save_path)
            # export val split
            save_path = str(Path(output_dir) / "val.json")
            save_json(result["val_coco"].json, save_path)
        else:
            save_path = str(Path(output_dir) / "dataset.json")
            save_json(coco.json, save_path)
            
class labelme2yolo:
    def __init__(self, dataset=None):
        print("ready to convert labelme data 2 yolo")
        self.dataset = dataset
        self.schema = [
            "img_folder",
            "img_filename",
            "img_path",
            "img_id",
            "img_width",
            "img_height",
            "img_depth",
            "ann_segmented",
            "ann_bbox_xmin",
            "ann_bbox_ymin",
            "ann_bbox_xmax",
            "ann_bbox_ymax",
            "ann_bbox_width",
            "ann_bbox_height",
            "ann_area",
            "ann_segmentation",
            "ann_iscrowd",
            "ann_pose",
            "ann_truncated",
            "ann_difficult",
            "cat_id",
            "cat_name",
            "cat_supercategory",
            "split",
            "annotated",
        ]

    def _ReindexCatIds(self, df, cat_id_index=0):
        """
        Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and
        then increment the cat_ids to index + number of categories.
        It's useful if the cat_ids are not continuous, especially for dataset subsets,
        or combined multiple datasets. Some models like Yolo require starting from 0 and others
        like Detectron require starting from 1.
        """
        assert isinstance(cat_id_index, int), "cat_id_index must be an int."

        # Convert empty strings to NaN and drop rows with NaN cat_id
        df_copy = df.replace(r"^\s*$", np.nan, regex=True)
        df_copy = df_copy[df.cat_id.notnull()]
        # Coerce cat_id to int
        df_copy["cat_id"] = pd.to_numeric(df_copy["cat_id"])

        # Map cat_ids to the range [cat_id_index, cat_id_index + num_cats)
        unique_ids = np.sort(df_copy["cat_id"].unique())
        ids_dict = dict((v, k) for k, v in enumerate(unique_ids, start=cat_id_index))
        df_copy["cat_id"] = df_copy["cat_id"].map(ids_dict)

        # Write back to the original dataframe
        df["cat_id"] = df_copy["cat_id"]
        
    def json2df(self, annotations_json, path, type=""):
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]        
        image_dir = None
        
        for file_path in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(file_path):
                for sub_path in glob.glob(os.path.join(file_path, '*')):
                    if "images/train" in sub_path:
                        break
                    elif "images/val" in sub_path:
                        break
                    elif os.path.isfile(sub_path) and any(sub_path.lower().endswith(ext) for ext in image_extensions):
                        image_dir = file_path
                        break
                    
        if type == "val":
            add_path = "/val"
        elif type == "train":
            add_path = "/train"
        else:
            add_path = ""
                    
        images = pd.json_normalize(annotations_json["images"])
        images.columns = "img_" + images.columns
        try:
            images["img_folder"]
        except:
            images["img_folder"] = ""
            # print(images)
                            
        astype_dict = {"img_width": "int64", "img_height": "int64", "img_depth": "int64"}
        astype_keys = list(astype_dict.keys())
        
        for element in astype_keys:
            if element not in images.columns:
                astype_dict.pop(element)
        # print(astype_dict)
        
        images = images.astype(astype_dict)
    
        annotations = pd.json_normalize(annotations_json["annotations"])
        annotations.columns = "ann_" + annotations.columns
    
        categories = pd.json_normalize(annotations_json["categories"])
        categories.columns = "cat_" + categories.columns
    
        # Converting this to string resolves issue #23
        categories.cat_id = categories.cat_id.astype(str)
                
        # Converting this to string resolves issue #23
        df = annotations
        df.ann_category_id = df.ann_category_id.astype(str)
    
        df[["ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_width", "ann_bbox_height"]] = pd.DataFrame(
            df.ann_bbox.tolist(), index=df.index
        )
        df.insert(8, "ann_bbox_xmax", df["ann_bbox_xmin"] + df["ann_bbox_width"])
        df.insert(10, "ann_bbox_ymax", df["ann_bbox_ymin"] + df["ann_bbox_height"])
    
        # debug print(df.info())
    
        # Join the annotions with the information about the image to add the image columns to the dataframe
        df = pd.merge(images, df, left_on="img_id", right_on="ann_image_id", how="left")
        df = pd.merge(df, categories, left_on="ann_category_id", right_on="cat_id", how="left")
        
        # Rename columns if needed from the coco column name to the pylabel column name
        df.rename(columns={"img_file_name": "img_filename"}, inplace=True)        
        df.rename(columns={"img_path": "img_path"}, inplace=True)
        df["img_path"] = add_path
        
        # Drop columns that are not in the schema
        df = df[df.columns.intersection(self.schema)]
    
        # Add missing columns that are in the schema but not part of the table
        df[list(set(self.schema) - set(df.columns))] = ""
    
        # Reorder columns
        df = df[self.schema]
        df.index.name = "id"
        df.annotated = 1
    
        # Fill na values with empty strings which resolved some errors when
        # working with images that don't have any annotations
        df.fillna("", inplace=True)
    
        # These should be strings
        df.cat_id = df.cat_id.astype(str)
    
        # These should be integers
        df.img_width = df.img_width.astype(int)
        df.img_height = df.img_height.astype(int)
    
        dataset = Dataset(df)
    
        # Assign the filename (without extension) as the name of the dataset
        dataset.name = Path(path).stem    
        dataset.path_to_annotations = PurePath(path).parent
        dataset.path_to_imgs = PurePath(image_dir)
    
        if self.dataset is not None:
            # Append the new dataset to the existing dataset
            self.dataset.df = pd.concat([self.dataset.df, dataset.df], ignore_index=True)
        else:
            self.dataset = dataset
        
    def extract_labelme(
        self,
        labelme_folder: str,
        category_list: List = None,
        train_split_rate: float = 1
    ) -> Coco:
        """
        Args:
            labelme_folder: folder that contains labelme annotations and image files
            category_list: start from a predefined coco cateory list
        """
        # get json list
        _, abs_json_path_list = list_files_recursively(
            labelme_folder, contains=[".json"]
        )
        labelme_json_list = abs_json_path_list
        labelme_json_list.sort()

        # init coco object
        coco = Coco()

        if category_list is not None:
            coco.add_categories_from_coco_category_list(category_list)

        # parse labelme annotations
        category_ind = 0
        for json_path in tqdm(
            labelme_json_list, "Extracting data from labelme json using sahi"
        ):
            try:
                data = load_json(json_path)
                
                # get image size
                image_path = str(Path(labelme_folder) / data["imagePath"])
                
                # use the image sizes provided by labelme (they already account for
                # things such as EXIF orientation)
                width = data["imageWidth"]
                height = data["imageHeight"]
                # init coco image
                coco_image = CocoImage(
                    file_name=data["imagePath"], height=height, width=width
                )
                # iterate over annotations
                for shape in data["shapes"]:
                    # set category name and id
                    category_name = shape["label"]
                    
                    category_id = None
                    for (
                        coco_category_id,
                        coco_category_name,
                    ) in coco.category_mapping.items():
                        if category_name == coco_category_name:
                            category_id = coco_category_id
                            break
                    # add category if not present
                    if category_id is None:
                        category_id = category_ind
                        coco.add_category(
                            CocoCategory(id=category_id, name=category_name)
                        )
                        category_ind += 1
    
                    # circles and lines to segmentation
                    if shape["shape_type"] == "circle":
                        (cx, cy), (x1, y1) = shape["points"]
                        r = np.linalg.norm(np.array([x1 - cx, y1 - cy]))
                        angles = np.linspace(0, 2 * np.pi, 50 * (int(r) + 1))
                        x = cx + r * np.cos(angles)
                        y = cy + r * np.sin(angles)
                        points = np.rint(np.append(x, y).reshape(-1, 2, order="F"))
                        _, index = np.unique(points, return_index=True, axis=0)
                        shape["points"] = points[np.sort(index)]
                        shape["shape_type"] = "polygon"
                    elif shape["shape_type"] == "line":
                        (x1, y1), (x2, y2) = shape["points"]
                        shape["points"] = [
                            x1,
                            y1,
                            x2,
                            y2,
                            x2 + 1e-3,
                            y2 + 1e-3,
                            x1 + 1e-3,
                            y1 + 1e-3,
                        ]
                        shape["shape_type"] = "polygon"
    
                    # parse bbox/segmentation
                    if shape["shape_type"] == "rectangle":
                        x1 = shape["points"][0][0]
                        y1 = shape["points"][0][1]
                        x2 = shape["points"][1][0]
                        y2 = shape["points"][1][1]
                        coco_annotation = CocoAnnotation(
                            bbox=[x1, y1, x2 - x1, y2 - y1],
                            category_id=category_id,
                            category_name=category_name,
                        )
                    elif shape["shape_type"] == "polygon":
                        segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                        coco_annotation = CocoAnnotation(
                            segmentation=segmentation,
                            category_id=category_id,
                            category_name=category_name,
                        )
                    else:
                        raise NotImplementedError(
                            f'shape_type={shape["shape_type"]} not supported.'
                        )
                    coco_image.add_annotation(coco_annotation)
                coco.add_image(coco_image)
            except Exception as e:
                continue
        
        if 0 < train_split_rate < 1:
            result = coco.split_coco_as_train_val(train_split_rate)
            self.json2df(result["train_coco"].json, labelme_folder, "train")
            self.json2df(result["val_coco"].json, labelme_folder, "val")
        else:
            self.json2df(coco.json, labelme_folder)                    
                                
    def convert(
        self,
        input_dir: str,
        output_dir: str,
        train_split_rate: float = 1,
        yaml_file="datasets.yaml",
        copy_images=False,
        use_splits=False,
        cat_id_index=None,
        segmentation=False,
    ):
        """Writes annotation files to disk in YOLOv5 format and returns the paths to files.

        Args:
            input_dir (str):
                This is where the raw dataset is stored.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object. If you are exporting images to train a model, the recommended path
                to use is 'training/labels'.
            output_dir (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object. If you are exporting images to train a model, the recommended path
                to use is 'training/labels'.
            yaml_file (str):
                If a file name (string) is provided, a YOLO YAML file will be created with entries for the files
                and classes in this dataset. It will be created in the parent of the output_dir directory.
                The recommended name for the YAML file is 'datasets.yaml'.
            copy_images (boolean):
                If True, then the annotated images will be copied to a directory next to the labels directory into
                a directory named 'images'. This will prepare your labels and images to be used as inputs to
                train a YOLOv5 model.
            use_splits (boolean):
                If True, then the images and annotations will be moved into directories based on the values in the split column.
                For example, if a row has the value split = "train" then the annotations for that row will be moved to directory
                /train. If a YAML file is specificied then the YAML file will use the splits to specify the folders user for the
                train, val, and test datasets.
            cat_id_index (int):
                Reindex the cat_id values so that that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Yolo requires the set of annotations to start at 0 when training a model.
            segmentation (boolean):
                If true, then segmentation annotations will be exported instead of bounding box annotations.
                If there are no segmentation annotations, then no annotations will be empty.

        Returns:
            A list with 1 or more paths (strings) to annotations files. If a YAML file is created
            then the first item in the list will be the path to the YAML file.

        Examples:
            >>> dataset.export.ExportToYolo(output_dir='training/labels',
            >>>     yaml_file='dataset.yaml', cat_id_index=0)
            ['training/dataset.yaml', 'training/labels/frame_0002.txt', ...]

        """
        self.extract_labelme(labelme_folder=input_dir, train_split_rate=train_split_rate)
        add_path = []
        ds = self.dataset        

        # Inspired by https://github.com/aws-samples/groundtruth-object-detection/blob/master/create_annot.py
        yolo_dataset = ds.df.copy(deep=True)
        
        # Convert nan values in the split column from nan to '' because those are easier to work with with when building paths
        yolo_dataset.split = yolo_dataset.split.fillna("")
        for data_path in yolo_dataset["img_path"]:
            if data_path not in add_path:
                add_path.append(data_path)
                
        # Create all of the paths that will be used to manage the files in this dataset
        path_dict = {}

        # The output path is the main path that will be used to create the other relative paths
        path = PurePath(output_dir)
        path_dict["label_path"] = output_dir
        
        # The /images directory should be next to the /labels directory
        path_dict["image_path"] = str(PurePath(path, "images"))
        
        # The root directory is in parent of the /labels and /images directories
        path_dict["root_path"] = str(PurePath(path.parent))
        
        # The YAML file should be in root directory
        path_dict["yaml_path"] = str(PurePath(path, yaml_file))
        
        # The root directory will usually be next to the yolov5 directory.
        # Specify the relative path
        path_dict["root_path_from_yolo_dir"] = str(PurePath("../"))
        
        # If these default values to not match the users environment then they can manually edit the YAML file
        if copy_images:
            # Create the folder that the images will be copied to
            for i in range(len(add_path)):
                Path(path_dict["image_path"]+add_path[i]).mkdir(parents=True, exist_ok=True)

        # Drop rows that are not annotated
        # Note, having zero annotates can still be considered annotated
        # in cases when are no objects in the image thats should be indentified
        yolo_dataset = yolo_dataset.loc[yolo_dataset["annotated"] == 1]

        # yolo_dataset["cat_id"] = (
        #     yolo_dataset["cat_id"].astype("float").astype(pd.Int32Dtype())
        # )

        yolo_dataset.cat_id = yolo_dataset.cat_id.replace(r"^\s*$", np.nan, regex=True)

        pd.to_numeric(yolo_dataset["cat_id"])

        if cat_id_index != None:
            assert isinstance(cat_id_index, int), "cat_id_index must be an int."
            self._ReindexCatIds(yolo_dataset, cat_id_index)

        # Convert empty bbox coordinates to nan to avoid math errors
        # If an image has no annotations then an empty label file will be created
        yolo_dataset.ann_bbox_xmin = yolo_dataset.ann_bbox_xmin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_ymin = yolo_dataset.ann_bbox_ymin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_width = yolo_dataset.ann_bbox_width.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_height = yolo_dataset.ann_bbox_height.replace(
            r"^\s*$", np.nan, regex=True
        )

        # If segmentation = False then export bounding boxes
        if segmentation == False:
            yolo_dataset["center_x_scaled"] = (
                yolo_dataset["ann_bbox_xmin"] + (yolo_dataset["ann_bbox_width"] * 0.5)
            ) / yolo_dataset["img_width"]
            yolo_dataset["center_y_scaled"] = (
                yolo_dataset["ann_bbox_ymin"] + (yolo_dataset["ann_bbox_height"] * 0.5)
            ) / yolo_dataset["img_height"]
            yolo_dataset["width_scaled"] = (
                yolo_dataset["ann_bbox_width"] / yolo_dataset["img_width"]
            )
            yolo_dataset["height_scaled"] = (
                yolo_dataset["ann_bbox_height"] / yolo_dataset["img_height"]
            )

        # Create folders to store annotations
        if output_dir == None:
            dest_folder = PurePath(ds.path_to_annotations, yolo_dataset.iloc[0].img_folder)
        else:
            dest_folder = str(PurePath(output_dir, "labels"))

        for i in range(len(add_path)):
            Path(dest_folder+add_path[i]).mkdir(parents=True, exist_ok=True)

        unique_images = yolo_dataset["img_filename"].unique()
        output_file_paths = []
        pbar = tqdm(desc="Converting labelme annotations to YOLO format", total=len(unique_images))
        for img_filename in unique_images:
            df_single_img_annots = yolo_dataset.loc[yolo_dataset.img_filename == img_filename]
            basename, _ = os.path.splitext(img_filename)
            annot_txt_file = basename + ".txt"
            # Use the value of the split collumn to create a directory
            # The values should be train, val, test or ''
            split_dir = df_single_img_annots["img_path"].iloc[0]
            annot_dir = dest_folder + split_dir
            destination = str(PurePath(annot_dir, annot_txt_file))
          
            # If segmentation = false then output bounding boxes
            if segmentation == False:
                df_single_img_annots.to_csv(
                    destination,
                    index=False,
                    header=False,
                    sep=" ",
                    float_format="%.4f",
                    columns=[
                        "cat_id",
                        "center_x_scaled",
                        "center_y_scaled",
                        "width_scaled",
                        "height_scaled",
                    ],
                )

            # If segmentation = true then output the segmentation mask
            else:
                # Create one file for image
                with open(destination, "w") as file:
                    # Create one row per row in the data frame
                    for i in range(0, df_single_img_annots.shape[0]):
                        row = str(df_single_img_annots.iloc[i].cat_id)
                        segmentation_array = df_single_img_annots.iloc[i].ann_segmentation[0]

                        # Iterate through every value of the segmentation array
                        # To normalize the coordinates from 0-1
                        for index, l in enumerate(segmentation_array):
                            # The first number in the array is the x value so divide by the width
                            if index % 2 == 0:
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_width
                                    )
                                )
                            else:
                                # The first number in the array is the x value so divide by the height
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_height
                                    )
                                )

                        file.write(row + "\n")

            output_file_paths.append(destination)

            if copy_images:
                source_image_path = str(
                    Path(
                        ds.path_to_imgs,
                        df_single_img_annots.iloc[0].img_folder,
                        df_single_img_annots.iloc[0].img_filename,
                    )
                )
                current_file = Path(source_image_path)
                assert (
                    current_file.is_file
                ), f"File does not exist: {source_image_path}. Check img_folder column values."
                labeled_path = path_dict["image_path"] + split_dir
                shutil.copy(
                    str(source_image_path),
                    str(PurePath(labeled_path, img_filename)),
                )
            pbar.update()

        # Create YAML file
        if yaml_file:
            # Make a set with all of the different values of the split column
            splits = set(yolo_dataset.split)
            # Build a dict with all of the values that will go into the YAML file
            dict_file = {}
            dict_file["path"] = path_dict["root_path_from_yolo_dir"]
          

            # Define the path for train/val data
            dict_file["train"] = path_dict["image_path"]
            dict_file["val"] = path_dict["image_path"]
          
            for i in range(len(add_path)):
                if add_path[i] == "/train":
                    dict_file["train"] = str(PurePath(path_dict["image_path"], "train"))
                elif add_path[i] == "/val":
                    dict_file["val"] = str(PurePath(path_dict["image_path"], "val"))

            dict_file["nc"] = ds.analyze.num_classes
            dict_file["names"] = ds.analyze.classes

            # Save the yamlfile
            with open(path_dict["yaml_path"], "w") as file:
                documents = yaml.dump(dict_file, file, encoding="utf-8", allow_unicode=True)                
                output_file_paths = [path_dict["yaml_path"]] + output_file_paths

        return output_file_paths