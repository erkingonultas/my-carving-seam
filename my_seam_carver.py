import numpy as np
import cv2
import sys

class SeamCarver:
    def __init__(self, filename, out_height, out_width, protect_mask='', object_mask='', progress_callback=None):
        # initialize parameter
        self.filename = filename
        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]
        # initialize parameter cont'd
        self.out_height = out_height
        self.out_width = out_width
        self.progress_callback = progress_callback


        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        # object removal --> self.object = True
        self.object = (object_mask != '')
        if self.object:
            # read in object mask image file as np.float64 format in gray scale
            self.mask = cv2.imread(object_mask, 0).astype(np.float64)
            self.protect = False
        # image re-sizing with or without protect mask
        else:
            self.protect = (protect_mask != '')
            if self.protect:
                # if protect_mask filename is provided, read in protect mask image file as np.float64 format in gray scale
                self.mask = cv2.imread(protect_mask, 0).astype(np.float64)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # constant for covered area by protect mask or object mask
        self.constant = 1000

        # starting program
        self.start()

    def _notify(self, message):
        """Emit progress updates to keep callers aware of long-running steps."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)

    def _notify_progress(self, prefix, current, total):
        """Update progress on the same terminal line so seam loops stay quiet."""
        if self.progress_callback:
            self.progress_callback(f"{prefix}: {current}/{total}")
            return

        end = "\n" if current == total else ""
        sys.stdout.write(f"\r{prefix}: {current}/{total}")
        sys.stdout.flush()
        if end:
            sys.stdout.write(end)

    def start(self):
        """
        :return:

        If object mask is provided --> object removal function will be executed
        else --> seam carving function (image retargeting) will be process
        """
        if self.object:
            self._notify("Starting object removal")
            self.object_removal()
        else:
            self._notify("Starting seam carving")
            self.seams_carving()

    def seams_carving(self):
        """
        :return:

        We first process seam insertion or removal in vertical direction then followed by horizontal direction.

        If targeting height or width is greater than original ones --> seam insertion,
        else --> seam removal

        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """

        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self._notify(f"Removing {abs(delta_col)} columns")
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self._notify(f"Inserting {delta_col} columns")
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self._notify(f"Removing {abs(delta_row)} rows")
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self._notify(f"Inserting {delta_row} rows")
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)

    def object_removal(self):
        """
        :return:

        Object covered by mask will be removed first and seam will be inserted to return to original image dimension
        """
        rotate = False
        object_height, object_width = self.get_object_dimension()
        if object_height < object_width:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.mask = self.rotate_mask(self.mask, 1)
            rotate = True

        while len(np.where(self.mask[:, :] > 0)[0]) > 0:
            self._notify("Carving seam over object region")
            energy_map = self.calc_energy_map()
            energy_map[np.where(self.mask[:, :] > 0)] *= -self.constant
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)
            self.delete_seam_on_mask(seam_idx)

        if not rotate:
            num_pixels = self.in_width - self.out_image.shape[1]
        else:
            num_pixels = self.in_height - self.out_image.shape[1]

        self._notify(f"Re-inserting {num_pixels} seams to restore size")
        self.seams_insertion(num_pixels)
        if rotate:
            self.out_image = self.rotate_image(self.out_image, 0)
    
    # --- Core vertical resizing primitive ---------------------------------
    #
    # `seams_removal` is the main driver for content-aware shrinking.
    #
    # For protected / object-masked images we still remove seams one by
    # one to respect the exact constraints. For the unmasked case we use
    # a *batched multi-seam* strategy:
    #   - Compute the energy and cumulative forward map once per batch.
    #   - Extract up to B disjoint seams from that DP map via
    #     `find_k_seams`.
    #   - Remove all of them in a single vectorized pass via
    #     `delete_seams_batch`.
    #
    # This reduces the number of full DP passes from O(K) to roughly
    # O(K / B), which is the key to making seam carving practical for
    # high-resolution images.
    def seams_removal(self, num_pixel):
        if self.protect:
            for dummy in range(num_pixel):
                self._notify_progress("Removing seams", dummy + 1, num_pixel)
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask > 0)] *= self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)
        else:
            # Batched multi-seam removal: remove several seams per DP pass
            remaining = num_pixel
            removed = 0
            while remaining > 0:
                # Limit batch size so we never remove too many seams relative to current width.
                current_width = self.out_image.shape[1]
                if current_width <= 2:
                    break

                # Heuristic: remove only a *small* number of seams per batch.
                #
                # If we remove too many seams from a single energy / DP map,
                # later seams will be chosen based on an outdated view of the
                # image and can start cutting through important structure.
                #
                # We therefore cap the batch size in two ways:
                #   - as a small fraction of the current width (e.g. 4%), and
                #   - by an absolute maximum (e.g. 8 seams per batch).
                #
                # This still reduces the number of full DP passes by roughly
                # a factor of 4–10 on large images, while keeping the visual
                # behaviour close to the classical one-seam-at-a-time method.
                max_batch_frac = 0.04   # at most 4% of current width per batch
                max_batch_cap = 64       # and never more than 8 seams at once
                max_batch = max(1, min(int(current_width * max_batch_frac), max_batch_cap))
                batch = min(remaining, max_batch)

                # Progress is tracked on total number of seams removed so far.
                self._notify_progress("Removing seams", removed + batch, num_pixel)

                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_forward(energy_map)
                seams = self.find_k_seams(cumulative_map, batch)

                if seams.size == 0:
                    break

                self.delete_seams_batch(seams)

                actual = seams.shape[0]
                remaining -= actual
                removed += actual


    def seams_insertion(self, num_pixel):
        if self.protect:
            temp_image = np.copy(self.out_image)
            temp_mask = np.copy(self.mask)
            seams_record = []

            for dummy in range(num_pixel):
                self._notify_progress("Recording seams for insertion (protected)", dummy + 1, num_pixel)
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask[:, :] > 0)] *= self.constant
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)

            self.out_image = np.copy(temp_image)
            self.mask = np.copy(temp_mask)
            n = len(seams_record)
            for dummy in range(n):
                self._notify_progress("Inserting seams", dummy + 1, n)
                seam = seams_record.pop(0)
                self.add_seam(seam)
                self.add_seam_on_mask(seam)
                seams_record = self.update_seams(seams_record, seam)
        else:
            # --- Batched seam insertion (unprotected case) -----------------
            #
            # We record seams to be inserted by repeatedly running a
            # *backward* dynamic program on the current image, but instead of
            # taking exactly one seam per DP pass, we take a small batch of
            # non-overlapping seams using `find_k_seams_backward`.
            #
            # Within each batch we still delete seams one by one and update
            # the remaining seams' indices so that the recorded sequence is
            # equivalent to the classic one-seam-at-a-time algorithm, while
            # requiring far fewer DP passes overall.
            temp_image = np.copy(self.out_image)
            seams_record = []
            remaining = num_pixel
            recorded = 0

            while remaining > 0:
                m, n = self.out_image.shape[:2]
                if n <= 2:
                    break

                # Conservative batch size: small fraction of width, capped.
                max_batch_frac = 0.02   # at most 4% of current width
                max_batch_cap = 8       # and never more than 8 seams per batch
                max_batch = max(1, min(int(n * max_batch_frac), max_batch_cap))
                batch = min(remaining, max_batch)

                # Backward DP for this batch.
                self._notify_progress(
                    "Recording seams for insertion (unprotected)",
                    min(recorded + batch, num_pixel),
                    num_pixel
                )
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_backward(energy_map)

                seams_batch = self.find_k_seams_backward(cumulative_map, batch)
                if seams_batch.size == 0:
                    break

                # Process each seam in the batch sequentially, updating the
                # remaining seams' indices after each deletion so that their
                # coordinates stay consistent with the current (shrinking)
                # image geometry.
                k_batch = seams_batch.shape[0]
                for i in range(k_batch):
                    seam = seams_batch[i]
                    seams_record.append(seam.copy())
                    self.delete_seam(seam)

                    # Update the remaining seams in this batch to account for
                    # the column shift caused by removing `seam`. For each
                    # later seam, any column index at or to the right of the
                    # removed pixel in a given row must shift left by 1.
                    if i + 1 < k_batch:
                        later = seams_batch[i + 1:]
                        # Vectorised per-row shift for all remaining seams.
                        # shape later: (k_remaining, m)
                        # broadcast seam (m,) against (k_remaining, m)
                        mask = later >= seam  # compare per row
                        later[mask] -= 1

                    recorded += 1
                    remaining -= 1
                    if remaining == 0:
                        break

            # Replay insertion on the original unmodified image. We restore
            # the original image and then insert seams in the same order they
            # were recorded. `update_seams` adjusts the indices of yet-to-be
            # inserted seams so that each new seam is applied to the current
            # (expanding) geometry.
            self.out_image = np.copy(temp_image)
            n = len(seams_record)
            for idx in range(n):
                self._notify_progress("Inserting seams", idx + 1, n)
                seam = seams_record.pop(0)
                self.add_seam(seam)
                seams_record = self.update_seams(seams_record, seam)


    def calc_energy_map(self):
        # Work in grayscale so we only run two Scharr operators instead of six channel-wise passes.
        gray = cv2.cvtColor(self.out_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return np.absolute(grad_x) + np.absolute(grad_y)


    def cumulative_map_backward(self, energy_map):
        """
        Classic backward cumulative energy map used for seam insertion.

        This is the standard dynamic program where the cost of a pixel
        depends only on the three pixels directly above it:
            up-left, up, up-right.

        We compute the DP row-by-row, but each row update is vectorised
        using shifted views of the previous row instead of an inner loop
        over columns. This significantly reduces Python overhead compared
        to the naive double loop, especially for wide images.
        """
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            prev = output[row - 1]

            # Cost coming from directly above
            e_up = prev

            # Cost coming from up-left: shift prev one to the right
            e_left = np.empty_like(prev)
            e_left[0] = np.inf
            e_left[1:] = prev[:-1]

            # Cost coming from up-right: shift prev one to the left
            e_right = np.empty_like(prev)
            e_right[-1] = np.inf
            e_right[:-1] = prev[1:]

            best = np.minimum(e_up, np.minimum(e_left, e_right))
            output[row] = energy_map[row] + best

        return output


    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            prev = output[row - 1]
            mx_prev = matrix_x[row - 1]
            myl_prev = matrix_y_left[row - 1]
            myr_prev = matrix_y_right[row - 1]

            e_up = prev + mx_prev

            e_left = np.empty_like(e_up)
            e_left[0] = np.inf
            e_left[1:] = prev[:-1] + mx_prev[:-1] + myl_prev[:-1]

            e_right = np.empty_like(e_up)
            e_right[-1] = np.inf
            e_right[:-1] = prev[1:] + mx_prev[1:] + myr_prev[1:]

            best = np.minimum(e_up, np.minimum(e_left, e_right))  # Vectorizes per-row DP step.
            output[row] = energy_map[row] + best
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output

    def find_k_seams(self, cumulative_map, k):
        """Find up to k non-overlapping seams from a single cumulative map.

        This function performs *one* dynamic-programming pass (already done
        in `cumulative_map_forward`) and then extracts multiple seams by
        repeated backtracking on a *modified* copy of the cumulative map.

        After extracting each seam, its pixels are set to +inf in the
        working copy so that subsequent seams cannot reuse those pixels.
        This guarantees that, for every row, no two seams share the same
        column index. As a result, removing these seams in a batch produces
        a rectangular image with a well-defined new width.

        Parameters
        ----------
        cumulative_map : np.ndarray
            Forward-energy cumulative cost map of shape (m, n).
        k : int
            Maximum number of seams to extract.

        Returns
        -------
        seams : np.ndarray
            Array of shape (num_seams, m), where each row is a seam
            expressed as a 1D array of column indices for each row.
            num_seams may be smaller than k if we run out of valid paths.
        """
        m, n = cumulative_map.shape

        # Work on a copy so we can invalidate pixels without touching
        # the original cumulative map.
        cm = np.copy(cumulative_map)
        seams = []
        rows = np.arange(m)

        for _ in range(k):
            # 1) Start each seam from the minimum-cost pixel on the bottom row.
            start_col = np.argmin(cm[-1])
            if not np.isfinite(cm[-1, start_col]):
                # No finite cost remains on the bottom row: we cannot
                # extract any more valid seams.
                break

            seam = np.zeros(m, dtype=np.int32)
            seam[-1] = start_col

            valid = True
            # 2) Backtrack upwards, always moving to one of the three
            #    neighbors (up-left, up, up-right) with the lowest cost
            #    *that is still finite* in the modified cumulative map.
            for row in range(m - 2, -1, -1):
                prev_x = seam[row + 1]
                start = max(prev_x - 1, 0)
                end = min(prev_x + 2, n - 1)

                candidates = cm[row, start:end + 1]

                # Mask out invalid / already-used pixels. If all three
                # candidates are inf, we cannot continue this seam.
                finite_mask = np.isfinite(candidates)
                if not np.any(finite_mask):
                    valid = False
                    break

                # Choose the best among the finite candidates only.
                effective = np.full_like(candidates, np.inf)
                effective[finite_mask] = candidates[finite_mask]
                offset = np.argmin(effective)
                col = start + offset

                seam[row] = col

            if not valid:
                # We failed to build a full seam from bottom to top.
                # Stop extracting further seams: the remaining finite
                # structure of cm cannot support more disjoint seams.
                break

            seams.append(seam)

            # 3) Invalidate all pixels along this seam for future extractions
            #    so that subsequent seams do not reuse any of these pixels.
            cm[rows, seam] = np.inf

        if not seams:
            return np.empty((0, m), dtype=np.int32)

        return np.stack(seams, axis=0)

    def find_k_seams_backward(self, cumulative_map, k):
        """Find up to k non-overlapping seams from a backward DP map.

        This is the backward-DP analogue of `find_k_seams` and is used
        when we are recording seams for insertion. It assumes that
        `cumulative_map` has already been computed by
        `cumulative_map_backward` and then repeatedly backtracks seams,
        invalidating used pixels by setting them to +inf.
        """
        m, n = cumulative_map.shape
        cm = np.copy(cumulative_map)
        seams = []
        rows = np.arange(m)

        for _ in range(k):
            # Start from the best pixel on the bottom row in the current map
            start_col = np.argmin(cm[-1])
            if not np.isfinite(cm[-1, start_col]):
                break

            seam = np.zeros(m, dtype=np.int32)
            seam[-1] = start_col
            valid = True

            # Backtrack upwards through the three neighbors (up-left, up,
            # up-right), always choosing the finite neighbor with minimum
            # cumulative cost.
            for row in range(m - 2, -1, -1):
                prev_x = seam[row + 1]
                start = max(prev_x - 1, 0)
                end = min(prev_x + 2, n - 1)

                candidates = cm[row, start:end + 1]
                finite_mask = np.isfinite(candidates)
                if not np.any(finite_mask):
                    valid = False
                    break

                effective = np.full_like(candidates, np.inf)
                effective[finite_mask] = candidates[finite_mask]
                offset = np.argmin(effective)
                seam[row] = start + offset

            if not valid:
                break

            seams.append(seam)
            # Invalidate the pixels along this seam so future seams do not
            # reuse the same (row, col) positions.
            cm[rows, seam] = np.inf

        if not seams:
            return np.empty((0, m), dtype=np.int32)

        return np.stack(seams, axis=0)

    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        rows = np.arange(m)
        mask = np.ones((m, n), dtype=bool)
        mask[rows, seam_idx] = False  # Vectorized carve removes the seam for all channels in one pass.
        self.out_image = self.out_image[mask].reshape(m, n - 1, 3)

    def delete_seams_batch(self, seams):
        """Delete multiple seams in one vectorized pass.

        `seams` should be an array of shape (k, m), where each row is a seam.
        """
        if seams.size == 0:
            return

        m, n = self.out_image.shape[:2]
        k = seams.shape[0]
        if k >= n:
            raise ValueError("Cannot remove more seams than image width")

        rows = np.arange(m)
        mask = np.ones((m, n), dtype=bool)
        for seam in seams:
            # For each seam, mark exactly one pixel per row as False.
            # Because `find_k_seams` guarantees that no two seams share
            # the same (row, col), every row will end up with exactly
            # `k` False entries and `n - k` True entries.
            mask[rows, seam] = False

        # Applying a 2D boolean mask to a (m, n, 3) image collapses the
        # first two dimensions and keeps the channel dimension, producing
        # an array of shape (m * (n - k), 3). We then reshape it back
        # into (m, n - k, 3). If this reshape ever fails, it means that
        # some rows did not lose exactly `k` pixels, which would indicate
        # a bug in the multi-seam extraction logic.
        self.out_image = self.out_image[mask].reshape(m, n - k, 3)


    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, : col] = self.out_image[row, : col]
            if col == 0:
                p = np.average(self.out_image[row, col: col + 2], axis=0)
                output[row, col] = self.out_image[row, col]
                output[row, col + 1] = p
                output[row, col + 2:] = self.out_image[row, col + 1:]
            else:
                p = np.average(self.out_image[row, col - 1: col + 1], axis=0)
                output[row, col] = p
                output[row, col + 1:] = self.out_image[row, col:]
        self.out_image = np.copy(output)


    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output


    def rotate_image(self, image, ccw):
        # np.rot90 uses optimized strides and avoids Python loops.
        return np.rot90(image, k=1 if ccw else -1)


    def rotate_mask(self, mask, ccw):
        return np.rot90(mask, k=1 if ccw else -1)


    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        rows = np.arange(m)
        mask = np.ones((m, n), dtype=bool)
        mask[rows, seam_idx] = False
        self.mask = self.mask[mask].reshape(m, n - 1)


    def add_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : col] = self.mask[row, : col]
            if col == 0:
                p = np.average(self.mask[row, col: col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 2:] = self.mask[row, col + 1:]
            else:
                p = np.average(self.mask[row, col - 1: col + 1])
                output[row, col] = p
                output[row, col + 1:] = self.mask[row, col:]
        self.mask = np.copy(output)


    def get_object_dimension(self):
        rows, cols = np.where(self.mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width


    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
