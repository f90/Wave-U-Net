# -*- coding: utf-8 -*-

from multiprocessing import Process, Queue, Event
from numpy.random import randint, seed


class MultistreamCache():
    def __init__(self,
                 worker_method,
                 worker_options,  # replace at least this many entries on each cache update. Can be fractional
                 alpha_smoother=0.99):   # the higher the more temporally smoothed is the average_replacement_rate. Not very important

        self.num_workers = worker_options["num_workers"]
        self.worker_method = worker_method
        self.worker_options = worker_options
        self.cache_size = worker_options["cache_size"]
        self.min_replacement_rate = worker_options["min_replacement_rate"]
        self.alpha_smoother = alpha_smoother

        # Internal Data Structures
        self.communication_queue = Queue(maxsize=50)  #TODO  hardcoded for now
        self.worker_handles = []
        self.cache = [None] * self.cache_size
        self.idx_next_item_to_be_updated = 0
        self.average_replacement_rate = self.min_replacement_rate
        self.exit_flag = Event()
        self.exit_flag.clear()
        self.counter_cache_items_updated = 0

        # call seed if this is used from different threads / processes
        seed()

    def start_workers(self):
        for k in range(self.num_workers):
            p = Process(target=self.worker_method,
                        args=(self.communication_queue,
                              self.exit_flag,
                              self.worker_options))
            p.start()
            self.worker_handles.append(p)

        # Fill cache
        print('----- Filling cache (Size: {}) -------'.format(self.cache_size))
        for k in range(self.cache_size):
            self.update_next_cache_item(self.communication_queue.get())
        print('----- Cache Filled -------')

        # We reset the update counter when starting the workers
        self.counter_cache_items_updated = 0

    def stop_workers(self):
        # We just kill them assuming there is nothing to be shut down properly.
        # This is somewhat brutal but simplifies things a lot and is enough for now
        self.exit_flag.set()
        for worker in self.worker_handles:
            worker.join(timeout=3)
            worker.terminate()  # try harder to kill it off if necessary

    def update_next_cache_item(self, data):
        self.cache[self.idx_next_item_to_be_updated] = data
        self.idx_next_item_to_be_updated = (self.idx_next_item_to_be_updated + 1) % self.cache_size
        self.counter_cache_items_updated += 1

    def update_cache_from_queue(self):

        # Implements a minimum update rate in terms of an average
        # number of items that have to be replaced in a call to this
        # function. If the average is not achieved, this functions
        # blocks until the required number of items are replaced in the
        # cache.

        num_replacements_current = 0
        average_replacement_rate_prev = self.average_replacement_rate

        while True:
            average_replacement_rate_updated = (1-self.alpha_smoother) * num_replacements_current + self.alpha_smoother * average_replacement_rate_prev

            if (average_replacement_rate_updated >= self.min_replacement_rate): #TODO only have additional condition if we actually want to use all available data samples everytime? and self.communication_queue.empty())
                break
            if num_replacements_current == self.cache_size:  # entire cache replaced? Your IO is super fast!
                break

            #print('Loading new item into cache from data list starting with ' + self.worker_options["file_list"][0])
            self.update_next_cache_item(self.communication_queue.get())
            num_replacements_current += 1

        # Final update of self.num_replacements_smoothed
        self.average_replacement_rate = average_replacement_rate_updated

    def get_cache_item(self, idx):
        return self.cache[idx]

    def set_cache_item(self, idx, item):
        self.cache[idx] = item