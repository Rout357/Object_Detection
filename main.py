import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation 


#Function that detected object
def object_detection():
    segment_image = instance_segmentation()
    segment_image.load_model("C:\\Users\\Ruslan\\Desktop\\Python knowleges\\MyProjects\\GitHub\\Object_Detection\\mask_rcnn_coco.h5")
    
    #Pass the argument of the thing we want to recognize
    target_class = segment_image.select_target_classes(car=True)

    #add path to our image before and after treatment
    result = segment_image.segmentImage(
        image_path = "image_1.jpg",
        show_bboxes=True,
        segment_target_classes=target_class,
        output_image_name= "output.jpg"
    )

    objects_count = len(result[0]["scores"])
    print(f"Car found: {objects_count}")


def main():
    object_detection()



if __name__ == '__main__':
    main()