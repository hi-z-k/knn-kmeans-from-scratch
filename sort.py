import random

class Sort:
    @staticmethod
    def _rand(l, h):
        return random.randint(l, h)
    @staticmethod
    def selection(nums:list[int], transform=lambda x: x):
        n = len(nums)
        for i in range(n):
            min_pos = i
            for j in range(i + 1, n):
                if transform(nums[j]) < transform(nums[min_pos]):
                    min_pos = j
            nums[i], nums[min_pos] = nums[min_pos], nums[i]
    @staticmethod
    def quick_random(nums:list[int], transform=lambda x: x):
        def _partition(l, h):
            pivot_pos = Sort._rand(l, h)
            nums[l], nums[pivot_pos] = nums[pivot_pos], nums[l]
            pivot = transform(nums[l])
            i = l + 1
            j = h

            while i <= j:
                while i <= j and transform(nums[i]) <= pivot:
                    i += 1
                while i <= j and transform(nums[j]) > pivot:
                    j -= 1

                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
            nums[j], nums[l] = nums[l], nums[j]
            return j
        def _sort(l,h):
            if l < h:
                p_idx = _partition(l, h)
                _sort(l, p_idx - 1)
                _sort(p_idx + 1, h)
        _sort(0, len(nums) - 1)