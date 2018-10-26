import sys
import os
import subprocess


class Logger(object):
    def __init__(self, filepath=None, mode='w'):
        self.file = None
        self.filepath = filepath
        if filepath is not None:
            self.file = open(filepath, mode=mode, buffering=0)

    def __enter__(self):
        return self

    def log(self, msg, should_print=True):
        if should_print:
            print '[LOG] {}'.format(msg)
        if self.file is not None:
            self.file.write('{}\n'.format(msg))

    def write(self, msg):
        sys.__stdout__.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def log(logger, msg, should_print=True):
    if logger:
        logger.log(msg, should_print)
    else:
        if should_print:
            print msg


class Tee:
    def __init__(self, log_path):
        self.prev_stdout_descriptor = os.dup(sys.stdout.fileno())
        self.prev_stderr_descriptor = os.dup(sys.stderr.fileno())

        tee = subprocess.Popen(['tee', log_path], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    def close(self):
        os.dup2(self.prev_stdout_descriptor, sys.stdout.fileno())
        os.close(self.prev_stdout_descriptor)
        os.dup2(self.prev_stderr_descriptor, sys.stderr.fileno())
        os.close(self.prev_stderr_descriptor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
