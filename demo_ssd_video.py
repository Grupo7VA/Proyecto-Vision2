import time
import cv2
import gluoncv as gcv
import mxnet as mx

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
print(ctx)

net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True,ctx=ctx)

# Load the webcam handler
cap = cv2.VideoCapture("video.mp4")
time.sleep(1) ### letting the camera autofocus

while(cap.isOpened()):
    # Load frame from the camera
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Image pre-processing
    frame = mx.nd.array(frame).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

    # Run frame through network
    rgb_nd = rgb_nd.as_in_context(ctx)

    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)

    x=cv2.resize(img,dsize=(512,512))
    print("FPS: ", int(1.0 / (time.time() - start_time)))
    text=str(int(1.0 / (time.time() - start_time))) + " FPS"
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30,(30))
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 1
    cv2.putText(img,text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    gcv.utils.viz.cv_plot_image(img)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
cap.release()
cv2.destroyAllWindows()