<h3>Performing Camera Calibration</h4>
<h5>1. Set up</h2>
<p>Fix camera on optical table and fix curvature test block on optical table, as shown below. The optimal distance
between the test block and camera is {x mm}. </p>

<h5>2. Take calibration images </h5>
<p> Affix the PAF rail to a flat surface normal to the camera lens via suction. Take {n} images of the straight PAF rail.
At this point do not alter the location of the camera or the test block. Save images in './colour_shape_sensing/calbration_images/'
</p>

<h5>3. Run calibration code </h5>
<p>Run <strong>calibration_test.py</strong> on the images to extract pixels per mm for x and y.  </p>
