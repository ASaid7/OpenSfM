# pyre-unsafe
import itertools
import logging
import math
import queue
import threading
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from opensfm import bow, features, io, log, masking, pygeometry, upright
from opensfm.context import parallel_map
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


def run_features_processing(data: DataSetBase, images: List[str], force: bool) -> None:
    """Main entry point for running features extraction on a list of images."""
    # Check if we're using a GPU-based feature extractor
    feature_type = data.config["feature_type"].upper()
    is_gpu_feature = feature_type in ["DISK"] and data.config["use_gpu"]
    use_batching = is_gpu_feature and data.config.get("disk_batch_processing", True)
    
    default_queue_size = 10
    max_queue_size = 200

    mem_available = log.memory_available()
    processes = data.config["processes"]
    
    # For GPU features in batch mode, force single-process for stable GPU access
    if is_gpu_feature:
        if use_batching:
            logger.info(f"Using {feature_type} with GPU and batch processing - forcing single process mode for stability")
            processes = 1
        else:
            logger.info(f"Using {feature_type} with GPU - forcing single process mode for stability")
            processes = 1
    
    # Set up DISK thread lock if not already created
    global _disk_lock
    if not hasattr(features, '_disk_lock'):
        features._disk_lock = threading.RLock()
    
    # Rest of the original function...
    if mem_available:
        # Use 90% of available memory
        ratio_use = 0.9
        mem_available *= ratio_use
        logger.info(
            f"Planning to use {mem_available} MB of RAM for both processing queue and parallel processing."
        )

        # 50% for the queue / 50% for parallel processing
        expected_mb = mem_available / 2
        expected_images = min(
            max_queue_size, int(expected_mb / average_image_size(data))
        )
        processing_size = average_processing_size(data)
        logger.info(
            f"Scale-space expected size of a single image : {processing_size} MB"
        )
        processes = min(max(1, int(expected_mb / processing_size)), processes)
    else:
        expected_images = default_queue_size
    logger.info(
        f"Expecting to queue at most {expected_images} images while parallel processing of {processes} images."
    )

    process_queue = queue.Queue(expected_images)
    arguments: List[Tuple[str, Any]] = []

    if processes == 1:
        for image in images:
            counter = Counter()
            read_images(process_queue, data, [image], counter, 1, force)
            
            # Use batch processing if applicable
            if use_batching:
                run_detection_batch(process_queue, data.config)
            else:
                run_detection(process_queue)
                
            process_queue.get()
    else:
        counter = Counter()
        read_processes = data.config["read_processes"]
        if 1.5 * read_processes >= processes:
            read_processes = max(1, processes // 2)

        chunk_size = math.ceil(len(images) / read_processes)
        chunks_count = math.ceil(len(images) / chunk_size)
        read_processes = min(read_processes, chunks_count)

        expected: int = len(images)
        for i in range(read_processes):
            images_chunk = images[i * chunk_size : (i + 1) * chunk_size]
            arguments.append(
                (
                    "producer",
                    (process_queue, data, images_chunk, counter, expected, force),
                )
            )
        for _ in range(processes):
            # Choose the appropriate consumer based on batching option
            if use_batching:
                arguments.append(("batch_consumer", (process_queue, data.config)))
            else:
                arguments.append(("consumer", (process_queue)))
                
        parallel_map(process, arguments, processes, 1)

def average_image_size(data: DataSetBase) -> float:
    average_size_mb = 0
    for camera in data.load_camera_models().values():
        average_size_mb += camera.width * camera.height * 4 / 1024 / 1024
    return average_size_mb / max(1, len(data.load_camera_models()))


def average_processing_size(data: DataSetBase) -> float:
    processing_size = data.config["feature_process_size"]

    min_octave_size = 16  # from covdet.c
    octaveResolution = 3  # from covdet.c
    start_size = processing_size * processing_size * 4 / 1024 / 1024
    last_octave = math.floor(math.log2(processing_size / min_octave_size))

    total_size = 0
    for _ in range(last_octave + 1):
        total_size += start_size * octaveResolution
        start_size /= 2
    return total_size


def is_high_res_panorama(
    data: DataSetBase, image_key: str, image_array: np.ndarray
) -> bool:
    """Detect if image is a panorama."""
    exif = data.load_exif(image_key)
    if exif:
        camera = data.load_camera_models()[exif["camera"]]
        w, h = int(exif["width"]), int(exif["height"])
        exif_pano = pygeometry.Camera.is_panorama(camera.projection_type)
    elif image_array is not None:
        h, w = image_array.shape[:2]
        exif_pano = False
    else:
        return False
    return w == 2 * h or exif_pano


class Counter:
    """Lock-less counter from https://julien.danjou.info/atomic-lock-free-counters-in-python/
    that relies on the CPython impl. of itertools.count() that is thread-safe. Used, as for
    some reason, joblib doesn't like a good old threading.Lock (everything is stuck)
    """

    def __init__(self) -> None:
        self.number_of_read = 0
        self.counter = itertools.count()
        self.read_lock = threading.Lock()

    def increment(self) -> None:
        next(self.counter)

    def value(self) -> int:
        with self.read_lock:
            value = next(self.counter) - self.number_of_read
            self.number_of_read += 1
        return value


def process(args: Tuple[str, Any]) -> None:
    process_type, real_args = args
    if process_type == "producer":
        queue, data, images, counter, expected, force = real_args
        read_images(queue, data, images, counter, expected, force)
    elif process_type == "consumer":
        queue = real_args
        run_detection(queue)
    elif process_type == "batch_consumer":
        queue, config = real_args
        run_detection_batch(queue, config)


def read_images(
    queue: queue.Queue,
    data: DataSetBase,
    images: List[str],
    counter: Counter,
    expected: int,
    force: bool,
) -> None:
    full_queue_timeout = 600
    for image in images:
        logger.info(f"Reading data for image {image} (queue-size={queue.qsize()})")
        image_array = data.load_image(image)
        if data.config["features_bake_segmentation"]:
            segmentation_array = data.load_segmentation(image)
            instances_array = data.load_instances(image)
        else:
            segmentation_array, instances_array = None, None
        args = image, image_array, segmentation_array, instances_array, data, force
        queue.put(args, block=True, timeout=full_queue_timeout)
        counter.increment()
        if counter.value() == expected:
            logger.info("Finished reading images")
            queue.put(None)


def run_detection(queue: queue.Queue):
    while True:
        args = queue.get()
        if args is None:
            queue.put(None)
            break
        image, image_array, segmentation_array, instances_array, data, force = args
        detect(image, image_array, segmentation_array, instances_array, data, force)
        del image_array
        del segmentation_array
        del instances_array


def bake_segmentation(
    image: np.ndarray,
    points: np.ndarray,
    segmentation: Optional[np.ndarray],
    instances: Optional[np.ndarray],
    exif: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    exif_height, exif_width, exif_orientation = (
        exif["height"],
        exif["width"],
        exif.get("orientation", 1),
    )
    height, width = image.shape[:2]
    if exif_height != height or exif_width != width:
        logger.error(
            f"Image has inconsistent EXIF dimensions ({exif_width}, {exif_height}) and image dimensions ({width}, {height}). Orientation={exif_orientation}"
        )

    panoptic_data = [None, None]
    for i, p_data in enumerate([segmentation, instances]):
        if p_data is None:
            continue
        new_height, new_width = p_data.shape
        ps = upright.opensfm_to_upright(
            points[:, :2],
            width,
            height,
            exif_orientation,
            new_width=new_width,
            new_height=new_height,
        ).astype(int)
        # pyre-fixme[6]: For 2nd argument expected `None` but got
        #  `ndarray[typing.Any, typing.Any]`.
        panoptic_data[i] = p_data[ps[:, 1], ps[:, 0]]
    return tuple(panoptic_data)


def detect(
    image: str,
    image_array: np.ndarray,
    segmentation_array: Optional[np.ndarray],
    instances_array: Optional[np.ndarray],
    data: DataSetBase,
    force: bool = False,
) -> None:
    log.setup()

    need_words = (
        data.config["matcher_type"] == "WORDS"
        or data.config["matching_bow_neighbors"] > 0
    )
    has_words = not need_words or data.words_exist(image)
    has_features = data.features_exist(image)

    if not force and has_features and has_words:
        logger.info(
            "Skip recomputing {} features for image {}".format(
                data.feature_type().upper(), image
            )
        )
        return

    logger.info(
        "Extracting {} features for image {}".format(data.feature_type().upper(), image)
    )

    start = timer()

    p_unmasked, f_unmasked, c_unmasked = features.extract_features(
        image_array, data.config, is_high_res_panorama(data, image, image_array)
    )

    # Load segmentation and bake it in the data
    if data.config["features_bake_segmentation"]:
        exif = data.load_exif(image)
        s_unsorted, i_unsorted = bake_segmentation(
            image_array, p_unmasked, segmentation_array, instances_array, exif
        )
        p_unsorted = p_unmasked
        f_unsorted = f_unmasked
        c_unsorted = c_unmasked
    # Load segmentation, make a mask from it mask and apply it
    else:
        s_unsorted, i_unsorted = None, None
        fmask = masking.load_features_mask(data, image, p_unmasked)
        p_unsorted = p_unmasked[fmask]
        f_unsorted = f_unmasked[fmask]
        c_unsorted = c_unmasked[fmask]

    if len(p_unsorted) == 0:
        logger.warning("No features found in image {}".format(image))

    size = p_unsorted[:, 2]
    order = np.argsort(size)
    p_sorted = p_unsorted[order, :]
    f_sorted = f_unsorted[order, :]
    c_sorted = c_unsorted[order, :]
    if s_unsorted is not None:
        semantic_data = features.SemanticData(
            s_unsorted[order],
            i_unsorted[order] if i_unsorted is not None else None,
            data.segmentation_labels(),
        )
    else:
        semantic_data = None
    features_data = features.FeaturesData(p_sorted, f_sorted, c_sorted, semantic_data)
    data.save_features(image, features_data)

    if need_words:
        bows = bow.load_bows(data.config)
        n_closest = data.config["bow_words_to_match"]
        closest_words = bows.map_to_words(
            f_sorted, n_closest, data.config["bow_matcher_type"]
        )
        data.save_words(image, closest_words)

    end = timer()
    report = {
        "image": image,
        "num_features": len(p_sorted),
        "wall_time": end - start,
    }
    data.save_report(io.json_dumps(report), "features/{}.json".format(image))

def run_detection_batch(queue: queue.Queue, config: Dict[str, Any]):
    """Process images in batches for GPU-based feature extraction."""
    feature_type = config["feature_type"].upper()
    use_batching = config.get("disk_batch_processing", True)
    max_batch_size = config.get("disk_max_batch_size", 16)
    batch_timeout = config.get("disk_batch_timeout", 0.5)
    
    # Fall back to non-batched processing if conditions aren't met
    if not use_batching or feature_type != "DISK" or max_batch_size <= 1:
        run_detection(queue)
        return
    
    batch = []
    batch_data = []
    feature_counts = []
    
    logger.info(f"Starting batch processing with max size {max_batch_size}")
    
    def process_batch():
        if not batch:
            return
            
        logger.info(f"Processing batch of {len(batch)} images")
        start = timer()
        
        # Extract images for batch processing
        images_batch = [item[1] for item in batch_data]
        is_pano_batch = [is_high_res_panorama(item[4], item[0], item[1]) for item in batch_data]
        
        # Process batch
        results = features.extract_features_disk_batch(images_batch, config, feature_counts)
        
        # Save results for each image
        for i, ((image, _, segmentation_array, instances_array, data, force), result, is_pano) in enumerate(zip(batch_data, results, is_pano_batch)):
            if result is None:
                logger.warning(f"No features found for image {image}")
                continue
                
            p_unmasked, f_unmasked = result
            
            # Compute normalized features, colors, etc.
            h, w = images_batch[i].shape[:2]
            c_unmasked = np.zeros((len(p_unmasked), 3), dtype=np.float32)
            
            # Extract pixel colors at keypoint locations
            if len(p_unmasked) > 0:
                xs = p_unmasked[:, 0].round().astype(int)
                ys = p_unmasked[:, 1].round().astype(int)
                
                # Ensure indices are within bounds
                xs = np.clip(xs, 0, w-1)
                ys = np.clip(ys, 0, h-1)
                
                colors = images_batch[i][ys, xs]
                if images_batch[i].ndim == 2 or images_batch[i].shape[2] == 1:
                    colors = np.repeat(colors[:, np.newaxis], 3, axis=1)
                c_unmasked = colors
            
            # Handle segmentation if enabled
            if data.config["features_bake_segmentation"]:
                exif = data.load_exif(image)
                s_unsorted, i_unsorted = bake_segmentation(
                    images_batch[i], p_unmasked, segmentation_array, instances_array, exif
                )
                p_unsorted = p_unmasked
                f_unsorted = f_unmasked
                c_unsorted = c_unmasked
            else:
                s_unsorted, i_unsorted = None, None
                fmask = masking.load_features_mask(data, image, p_unmasked)
                p_unsorted = p_unmasked[fmask]
                f_unsorted = f_unmasked[fmask]
                c_unsorted = c_unmasked[fmask]

            # Sort by feature size
            size = p_unsorted[:, 2]
            order = np.argsort(size)
            p_sorted = p_unsorted[order, :]
            f_sorted = f_unsorted[order, :]
            c_sorted = c_unsorted[order, :]
            
            if s_unsorted is not None:
                semantic_data = features.SemanticData(
                    s_unsorted[order],
                    i_unsorted[order] if i_unsorted is not None else None,
                    data.segmentation_labels(),
                )
            else:
                semantic_data = None
                
            # Apply normalization
            p_sorted, f_sorted, c_sorted = features.normalize_features(p_sorted, f_sorted, c_sorted, w, h)
            
            # Save features data
            features_data = features.FeaturesData(p_sorted, f_sorted, c_sorted, semantic_data)
            data.save_features(image, features_data)

            # Handle BoW if needed
            need_words = (
                data.config["matcher_type"] == "WORDS"
                or data.config["matching_bow_neighbors"] > 0
            )
            if need_words:
                bows = bow.load_bows(data.config)
                n_closest = data.config["bow_words_to_match"]
                closest_words = bows.map_to_words(
                    f_sorted, n_closest, data.config["bow_matcher_type"]
                )
                data.save_words(image, closest_words)

            end = timer()
            report = {
                "image": image,
                "num_features": len(p_sorted),
                "wall_time": end - start,
            }
            data.save_report(io.json_dumps(report), "features/{}.json".format(image))
        
        logger.info(f"Completed batch processing in {timer() - start:.2f}s")
    
    while True:
        try:
            # Try to get item with timeout
            try:
                args = queue.get(timeout=batch_timeout)
            except queue.Empty:
                # Process batch if we timed out with items
                if batch:
                    process_batch()
                    batch = []
                    batch_data = []
                    feature_counts = []
                continue
            
            # Check for end signal
            if args is None:
                if batch:
                    process_batch()
                queue.put(None)  # Pass end signal to next worker
                break
            
            # Extract arguments
            image, image_array, segmentation_array, instances_array, data, force = args
            
            # Check if we should process this image
            need_words = (
                data.config["matcher_type"] == "WORDS"
                or data.config["matching_bow_neighbors"] > 0
            )
            has_words = not need_words or data.words_exist(image)
            has_features = data.features_exist(image)

            if not force and has_features and has_words:
                logger.info(
                    "Skip recomputing {} features for image {}".format(
                        data.feature_type().upper(), image
                    )
                )
                queue.task_done()
                continue
                
            # Add to batch
            batch.append(image)
            batch_data.append((image, image_array, segmentation_array, instances_array, data, force))
            
            # Calculate feature count based on config settings
            is_panorama = is_high_res_panorama(data, image, image_array)
            features_count = (
                data.config["feature_min_frames_panorama"]
                if is_panorama
                else data.config["feature_min_frames"]
            )
            feature_counts.append(features_count)
            
            # Process if batch is full
            if len(batch) >= max_batch_size:
                process_batch()
                batch = []
                batch_data = []
                feature_counts = []
                
            queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            if args is not None:
                # Process this image individually as a fallback
                try:
                    detect(*args)
                except Exception as inner_e:
                    logger.error(f"Also failed to process individually: {inner_e}")
                queue.task_done()