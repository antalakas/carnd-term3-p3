from styx_msgs.msg import TrafficLight
import rospy
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from PIL import Image
import time


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ssd_inception_sim_model = os.path.join(dir_path, 'frozen_models/frozen_sim_inception/frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(dir_path, 'tl/label_map.pbtxt')
        # rospy.loginfo(PATH_TO_LABELS)
        NUM_CLASSES = 4

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # rospy.loginfo(self.category_index)

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(ssd_inception_sim_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction

        traffic_light_color = TrafficLight.UNKNOWN

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # ------------------------------------- Image identification -------------------------------------------
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

                image_np = np.asarray(image)
                rospy.loginfo(image_np.shape)
                # return TrafficLight.UNKNOWN

                # image_np = self.load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                time0 = time.time()

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                time1 = time.time()

                rospy.loginfo('------------------------------')
                rospy.loginfo("Time in milliseconds: " + str((time1 - time0) * 1000))

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                min_score_thresh = .50
                for i in range(boxes.shape[0]):
                    if scores is None or scores[i] > min_score_thresh:
                        class_name = self.category_index[classes[i]]['name']

                        if class_name == 'Red':
                            if scores[i] >= 0.5:
                                traffic_light_color = TrafficLight.RED

                        if class_name == 'Green':
                            if scores[i] >= 0.5:
                                traffic_light_color = TrafficLight.GREEN

                        if class_name == 'Yellow':
                            if scores[i] >= 0.5:
                                traffic_light_color = TrafficLight.YELLOW

                        if class_name == 'off':
                                traffic_light_color = TrafficLight.UNKNOWN

                        # rospy.loginfo('class name: ' + class_name + ", score: " + str(scores[i]))

        return traffic_light_color

    # def load_image_into_numpy_array(self, image):
    #     (im_width, im_height) = image.size
    #     return np.array(image.getdata()).reshape(
    #         (im_height, im_width, 3)).astype(np.uint8)