# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.data.multimodal import get_multimodal_settings, resolve_paired_image_path, split_modalities
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
#################################################################################################################
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        ir_result = []
        multimodal = get_multimodal_settings(self.args)
        assert self.batch is not None and self.model is not None
        batch_paths = self.batch[0]
        model_names = self.model.names
        for i, pred in enumerate(preds):
            img_path = batch_paths[i]
            orig_img = orig_imgs[i]
            if orig_img.ndim == 3 and orig_img.shape[-1] == 6 and multimodal["input_modality"] == "multimodal":
                visible_img, infrared_img = split_modalities(orig_img, multimodal["channel_order"])
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], visible_img.shape)
                results.append(Results(visible_img, path=img_path, names=model_names, boxes=pred))
                ir_path = resolve_paired_image_path(img_path, multimodal)
                ir_result.append(Results(infrared_img, path=str(ir_path), names=model_names, boxes=pred))
            else:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                results.append(Results(orig_img, path=img_path, names=model_names, boxes=pred))
        return results, ir_result
