from . import cv2
from . import np

from modules.im_trans import img_to_gray
from modules.im_trans import draw_ellipse

class GrabCut (object):
    """GrabCut.
    Args:
        height -- input FRAME height
        width -- input FRAME width
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        x = width // 2                           # Frame centre x
        y = height // 2                          # Frame centre y
        a = x // 2
        self.gc_rect = (x - a, y - a, x, y)      # GrabCut working rectangle

    def gc_mask(self, img, iters=2):
        """GrabCut image segmentation. Background identification.
        Args:
            img -- image (frame) to processing
            rect -- rectangular area to be segmented. Tuple (x, y, w, h)
            iters -- algorithm iterations
        Returns:
            gc_mask -- mask of foreground
        """
        # Create additional args required for GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        height, width = img.shape[:2]
        # GrabCut uses downsampled image. (cv2.pyrDown == x0.5)
        # All the dimensions scaled down by scale_factor.
        scale_factor = 2
        mask = np.zeros((height // scale_factor, width // scale_factor), np.uint8)
        cv2.grabCut(img=cv2.pyrDown(img), \
                    mask=mask, \
                    rect=tuple(x // scale_factor for x in self.gc_rect), \
                    bgdModel=bgdModel, 
                    fgdModel=fgdModel, 
                    iterCount=iters, \
                    mode=cv2.GC_INIT_WITH_RECT)
        # Substitutes all bg pixels(0,2) with sure background (0)
        gc_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') 
        return cv2.pyrUp(gc_mask)


class FaceDetection(object):
    """Face Detection.
    Args:
        height -- input FRAME height
        width -- input FRAME width
    """
    # Features for CascadeClassifier (frontal face)
    HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

    def __init__(self, height, width):
        self.height = height
        self.width = width
        x = width // 2                           # Frame centre x
        y = height // 2                          # Frame centre y
        a = x // 2
        # Faces detected (by default - rect in the centre)
        self.faces = [np.array([x - a, y - a, x, y])]
        self.gc_rect = (x - a, y - a, x, y)      # GrabCut working rectangle
        self.def_face = self.faces[:1]           # Default value (rect in centre)
        self._face_cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)

    def track_faces(self, img):
        """Apply oval mask over faces detected in the image.
        Args:
            img -- image
        Updates:
            self.faces -- list of faces detected
            self.gc_rect -- coords. tuple for GrabCut algorithm (x, y, w, h)
        Returns:
            fgmask -- binary mask with faces highlighted with oval
        """
        faces = self._detect_faces(img)
        if faces != []:                              # Face coords detected
            self.faces = faces
            self.gc_rect = self._faces_to_gc_rect(faces)
        else:                                        # Default values
            self.faces = self.def_face
            self.gc_rect = self._faces_to_gc_rect(self.faces, 1)
        fgmask = np.zeros((self.height, self.width, 3), np.uint8)
        fgmask = draw_ellipse(fgmask, self.faces)
        fgmask = img_to_gray(fgmask)
        return fgmask

    def _detect_faces(self, img):
        """Detects faces on the image.
        Args:
            img -- image
        Returns:
            faces -- LIST of np.array([x,y,w,h]) face coordinates
        """
        if not self._face_cascade.empty():
            gray = img_to_gray(img)
            faces = self._face_cascade.detectMultiScale(gray, \
                                                scaleFactor=1.3, \
                                                minNeighbors=4, \
                                                minSize=(30, 30), \
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) == 0:
                return []
            return faces

    @staticmethod
    def _faces_to_gc_rect(faces, default_val=0):
        """Scale up first found face rectangle by 2 times. To make GrabCut
        working area BIGGER.
        Args:
            faces -- LIST of np.array([x,y,w,h]) coords of faces detected
            default_val -- Flag: process faces list or use def. value instead
        Returns:
            (x, y, w, h) -- scaled up coords in tuple format
        """
        if not default_val:
            M = np.array([[1, 0, -0.5, 0],              # Scale matrix
                          [0, 1, 0 , -0.5],
                          [0, 0, 2, 0],
                          [0, 0, 0, 2]])
            v = faces[0]                                # Face coords
            scaled_rect = np.inner(M, v).astype(int)    # Inner product M * v    
            return tuple(x for x in scaled_rect)        # Convert to tuple
        return tuple(x for x in faces[0])


class MOG2(object):
    """Background Subtraction.
    !!!Not used in current implementation of Ghost class.!!!
    """
    def __init__(self):
        self._bs_mog2 = cv2.createBackgroundSubtractorMOG2(history=1000,\
                                                      varThreshold=25,\
                                                      detectShadows=False)
        self.learningRate = 0.02
        self.kernel_size = 5
        self.iters = 4

    def substract_bg(self, frame):
        """Apply Background Substraction from frame.
        Args:
            frame -- current frame
        Returns: 
            fgmask -- foreground mask
        """
        # Get FGMASK with MOG2
        gray = img_to_gray(frame)
        fgmask = self._bs_mog2.apply(gray, learningRate=self.learningRate)
        # Elliptical Kernel for morphology func
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,( \
                                self.kernel_size, self.kernel_size))
        # Open (remove white points from the background)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # Close (remove black points from the object)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # Dilation alg (increases white regions size)
        fgmask = cv2.dilate(fgmask, kernel, iterations=self.iters)        
        return fgmask