import argparse
import tensorflow as tf


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_data_files(data_sources):
    """Get data_files from data_sources.

    Args:
      data_sources: a list/tuple of files or the location of the data, i.e.
        /path/to/train@128, /path/to/train* or /tmp/.../train*

    Returns:
      a list of data_files.

    Raises:
      ValueError: if not data files are not found

    """
    if isinstance(data_sources, (list, tuple)):
        data_files = []
        for source in data_sources:
            data_files += get_data_files(source)
    else:
        if '*' in data_sources or '?' in data_sources or '[' in data_sources:
            data_files = tf.gfile.Glob(data_sources)
        else:
            data_files = [data_sources]
    if not data_files:
        raise ValueError('No data files found in %s' % (data_sources,))
    return data_files
