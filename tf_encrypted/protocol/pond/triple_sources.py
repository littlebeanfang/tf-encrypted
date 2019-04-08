import os
import random

import tensorflow as tf


class OnlineTripleSource:

    def __init__(self, device_name):
        self.device_name = device_name

    def _share(self, secret):

        with tf.name_scope("share"):
            share0 = secret.factory.sample_uniform(secret.shape)
            share1 = secret - share0

            # randomized swap to distribute who gets the seed
            if random.random() < 0.5:
                share0, share1 = share1, share0

        return share0, share1

    def mul_triple(self, a, b):

        with tf.device(self.device_name):
            with tf.name_scope("triple"):
                ab = a * b
                ab0, ab1 = self._share(ab)

        return ab0, ab1

    def square_triple(self, a):

        with tf.device(self.device_name):
            with tf.name_scope("triple"):
                aa = a * a
                aa0, aa1 = self._share(aa)

        return aa0, aa1

    def matmul_triple(self, a, b):

        with tf.device(self.device_name):
            with tf.name_scope("triple"):
                ab = a.matmul(b)
                ab0, ab1 = self._share(ab)

        return ab0, ab1

    def conv2d_triple(self, a, b, strides, padding):

        with tf.device(self.device_name):
            with tf.name_scope("triple"):
                a_conv2d_b = a.conv2d(b, strides, padding)
                a_conv2d_b0, a_conv2d_b1 = self._share(a_conv2d_b)

        return a_conv2d_b0, a_conv2d_b1

    def indexer_mask(self, a, slice):

        with tf.device(self.device_name):
            a_sliced = a[slice]

        return a_sliced

    def transpose_mask(self, a, perm):

        with tf.device(self.device_name):
            a_t = a.transpose(perm=perm)

        return a_t

    def strided_slice_mask(self, a, args, kwargs):

        with tf.device(self.device_name):
            a_slice = a.strided_slice(args, kwargs)

        return a_slice

    def split_mask(self, a, num_split, axis):

        with tf.device(self.device_name):
            bs = a.split(num_split=num_split, axis=axis)

        return bs

    def stack_mask(self, bs, axis):

        factory = bs[0].factory

        with tf.device(self.device_name):
            b_stacked = factory.stack(bs, axis=axis)

        return b_stacked

    def concat_mask(self, bs, axis):

        factory = bs[0].factory

        with tf.device(self.device_name):
            b_stacked = factory.concat(bs, axis=axis)

        return b_stacked

    def reshape_mask(self, a, shape):

        with tf.device(self.device_name):
            a_reshaped = a.reshape(shape=shape)

        return a_reshaped

    def expand_dims_mask(self, a, axis):

        with tf.device(self.device_name):
            a_e = a.expand_dims(axis=axis)

        return a_e

    def squeeze_mask(self, a, axis):

        with tf.device(self.device_name):
            a_squeezed = a.squeeze(axis=axis)

        return a_squeezed

    def mask(self, backing_dtype, shape):

        with tf.device(self.device_name):
            a0 = backing_dtype.sample_uniform(shape)
            a1 = backing_dtype.sample_uniform(shape)
            a = a0 + a1

        return a, a0, a1

    def cache(self, a):
        raise NotImplementedError()

    def initialize(self, sess, tag=None):
        pass

    def generate_triples(self, sess, num=None, tag=None):
        pass


class QueuedTripleSource:

    # TODO(Morten) manually unwrap and re-wrap of queued values, should be hidden away

    def __init__(self, player0, player1, producer, capacity=10):
        self.player0 = player0
        self.player1 = player1
        self.producer = producer
        self.capacity = capacity
        self.queues = list()
        self.enqueuers = list()
        self.sizes = list()

    def mask(self, backing_dtype, shape):

        with tf.name_scope("triple-generation"):
            with tf.device(self.producer.device_name):
                a0 = backing_dtype.sample_uniform(shape)
                a1 = backing_dtype.sample_uniform(shape)
                a = a0 + a1

        d0, d1 = self._build_queues(a0, a1, "mask-queue")
        return a, d0, d1

    def mul_triple(self, a, b):

        with tf.name_scope("triple-generation"):
            with tf.device(self.producer.device_name):
                ab = a * b
                ab0, ab1 = self._share(ab)

        return self._build_queues(ab0, ab1, "triple-queue")

    def square_triple(self, a):

        with tf.name_scope("triple-generation"):
            with tf.device(self.producer.device_name):
                aa = a * a
                aa0, aa1 = self._share(aa)

        return self._build_queues(aa0, aa1, "triple-queue")

    def matmul_triple(self, a, b):

        with tf.name_scope("triple-generation"):
            with tf.device(self.producer.device_name):
                ab = a.matmul(b)
                ab0, ab1 = self._share(ab)
    
        return self._build_queues(ab0, ab1, "triple-queue")

    def conv2d_triple(self, a, b, strides, padding):

        with tf.device(self.producer.device_name):
            with tf.name_scope("triple"):
                ab = a.conv2d(b, strides, padding)
                ab0, ab1 = self._share(ab)

        return self._build_queues(ab0, ab1)

    def indexer_mask(self, a, slice):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_sliced = a[slice]

        return a_sliced

    def transpose_mask(self, a, perm):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_t = a.transpose(perm=perm)

        return a_t

    def strided_slice_mask(self, a, args, kwargs):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_slice = a.strided_slice(args, kwargs)

        return a_slice

    def split_mask(self, a, num_split, axis):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                bs = a.split(num_split=num_split, axis=axis)

        return bs

    def stack_mask(self, bs, axis):

        factory = bs[0].factory

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                b_stacked = factory.stack(bs, axis=axis)

        return b_stacked

    def concat_mask(self, bs, axis):

        factory = bs[0].factory

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                b_stacked = factory.concat(bs, axis=axis)

        return b_stacked

    def reshape_mask(self, a, shape):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_reshaped = a.reshape(shape)

        return a_reshaped

    def expand_dims_mask(self, a, axis):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_e = a.expand_dims(axis=axis)

        return a_e

    def squeeze_mask(self, a, axis):

        with tf.name_scope("mask-transformation"):
            with tf.device(self.producer.device_name):
                a_squeezed = a.squeeze(axis=axis)

        return a_squeezed

    def cache(self, a):
        raise NotImplementedError()

    def _share(self, secret):

        with tf.name_scope("share"):
            share0 = secret.factory.sample_uniform(secret.shape)
            share1 = secret - share0

            # randomized swap to distribute who gets the seed
            if random.random() < 0.5:
                share0, share1 = share1, share0

        return share0, share1

    def _build_queues(self, c0, c1, queue_name=None):

        def build_triple_store(mask):

            raw_mask = mask.value
            factory = mask.factory
            dtype = mask.factory.native_type
            shape = mask.shape

            with tf.name_scope("triple-store"):

                q = tf.queue.FIFOQueue(
                    capacity=self.capacity,
                    dtypes=[dtype],
                    shapes=[shape],
                )
                e = q.enqueue(raw_mask)
                d = factory.tensor(q.dequeue())

            return q, e, d

        with tf.device(self.player0.device_name):
            q0, e0, d0 = build_triple_store(c0)

        with tf.device(self.player1.device_name):
            q1, e1, d1 = build_triple_store(c1)

        self.queues += [q0, q1]
        self.enqueuers += [e0, e1]
        return d0, d1

    def initialize(self, sess, tag=None):
        pass

    def generate_triples(self, sess, num=1, tag=None):
        for _ in range(num):
            sess.run(self.enqueuers, tag=tag)

