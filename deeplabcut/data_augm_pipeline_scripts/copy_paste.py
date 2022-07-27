#/home/sofia/miniconda/envs/deeplabcut-res/lib/python3.8/site-packages/imgaug/augmenters/size.py
import imgaug as ia
import imgaug.augmenters as iaa  #from imgaug.augmenters import Augmenter
import numpy as np

class CopyPaste(iaa.meta.Augmenter):
    ############################################################################
    def __init__(self, p=1,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # parent class
        super(CopyPaste, self).__init__(seed=seed, 
                                        name=name,
                                        random_state=random_state, 
                                        deterministic=deterministic)
        self.p = ia.parameters.handle_probability_param(p, "p")

    #############################################################################
    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        samples = self.p.draw_samples((batch.nb_rows,),
                                      random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                if batch.images is not None:
                    # We currently do not use flip.flipud() here, because that
                    # saves a function call.
                    batch.images[i] = batch.images[i][::-1, ...]

                if batch.heatmaps is not None:
                    batch.heatmaps[i].arr_0to1 = \
                        batch.heatmaps[i].arr_0to1[::-1, ...]

                if batch.segmentation_maps is not None:
                    batch.segmentation_maps[i].arr = \
                        batch.segmentation_maps[i].arr[::-1, ...]

                if batch.keypoints is not None:
                    kpsoi = batch.keypoints[i]
                    height = kpsoi.shape[0]
                    for kp in kpsoi.keypoints:
                        kp.y = height - float(kp.y)

                if batch.bounding_boxes is not None:
                    bbsoi = batch.bounding_boxes[i]
                    height = bbsoi.shape[0]
                    for bb in bbsoi.bounding_boxes:
                        # after flip, y1 ends up right of y2
                        y1, y2 = bb.y1, bb.y2
                        bb.y1 = height - y2
                        bb.y2 = height - y1

                if batch.polygons is not None:
                    psoi = batch.polygons[i]
                    height = psoi.shape[0]
                    for poly in psoi.polygons:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        poly.exterior[:, 1] = height - poly.exterior[:, 1]

                if batch.line_strings is not None:
                    lsoi = batch.line_strings[i]
                    height = lsoi.shape[0]
                    for ls in lsoi.line_strings:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        ls.coords[:, 1] = height - ls.coords[:, 1]

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]
