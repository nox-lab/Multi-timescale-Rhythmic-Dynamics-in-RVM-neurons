"""
Class for reading data from CED (Cambridge Electronic Design)
http://ced.co.uk/

This allows the reading in of smrx files, which in the orignal neo class
`neo.rawio.cedrawio` cannot be read.

For more details see:
https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/spike2rawio.py

and:
https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/cedrawio.py

Author : Carl Ashworth, adapted from original by Samuel Garcia
"""

from neo.rawio.baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

import numpy as np
from datetime import datetime


class CedRawIO(BaseRawIO):
    """
    Class for reading data from CED (Cambridge Electronic Design) spike2.
    This internally uses the sonpy package which is closed source.

    This IO reads smr and smrx files
    """

    extensions = ["smr", "smrx"]
    rawmode = "one-file"

    def __init__(
        self,
        filename="",
        take_ideal_sampling_rate=False,
    ):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.max_time_overall = 0
        self.take_ideal_sampling_rate = take_ideal_sampling_rate
        self._all_event_labels = {}


    def _source_name(self):
        return self.filename
    
    def _parse_to_datetime(self,values):
        """
        values = [minutes, seconds, milliseconds, hours, day, month, year]
        """
        print("time values", values)
        subsecond, second, minute, hour, day, month, year = values

        return datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        )

    def _parse_header(self):

        import sonpy
        sp = sonpy.lib

        self.smrx_file = sp.SonFile(sName=str(self.filename), bReadOnly=True)
        print(self.smrx_file)
        self._time_base = self.smrx_file.GetTimeBase()
        print("The start of sampling was: ", self._parse_to_datetime(self.smrx_file.GetTimeDate()))

        self._rec_datetime = self._parse_to_datetime(
            self.smrx_file.GetTimeDate()
        )
        channel_infos = []
        signal_channels = []
        spike_channels = []
        event_channels = []
        self._all_spike_ticks = {}
        self._all_event_ticks = {}
        self._all_signals = {}
        n_channels = self.smrx_file.MaxChannels()

        self.max_time_overall = self.smrx_file.MaxTime()

        for ch_ind in range(n_channels):
            ch_type = self.smrx_file.ChannelType(ch_ind)
            if (
                ch_type != sp.DataType.Off
            ):
                try:
                    ch_name = self.smrx_file.GetChannelTitle(ch_ind)
                except ValueError:
                    print(
                        "Channel Not Named - edit this code "
                        "to add a name for the channel here."
                    )
                print(f"    {ch_type}: {ch_ind}, {ch_name}")

                max_time = self.smrx_file.ChannelMaxTime(ch_ind)
                divide = self.smrx_file.ChannelDivide(ch_ind)
                sampling_rate = 1 / (divide * self._time_base)

                if ch_type == sp.DataType.AdcMark:
                    wave_marks = self.smrx_file.ReadWaveMarks(
                        ch_ind, int(max_time / divide), 0, max_time + 1
                    )

                    spike_ticks = np.array([t.Tick for t in wave_marks])
                    spike_codes = np.array([t.Code1 for t in wave_marks])

                    unit_ids = np.unique(spike_codes)
                    for unit_id in unit_ids:
                        print(f"Saving the unit with id {unit_id}, channel ind{ch_ind}")
                        name = f"{ch_name}#{unit_id}"
                        spike_ch_id = f"ch{ch_ind}#{unit_id}"
                        spike_channels.append((name, spike_ch_id, "", 1, 0, 0, 0))
                        mask = spike_codes == unit_id
                        self._all_spike_ticks[spike_ch_id] = spike_ticks[mask]

                elif ch_type in [
                    sp.DataType.EventRise,
                    sp.DataType.EventFall,
                    sp.DataType.EventBoth,
                ]:
                    events = np.array(
                        self.smrx_file.ReadEvents(
                            ch_ind,
                            int(max_time * self._time_base) * 15,
                            0,
                            max_time + 1,
                        )
                    )

                    event_ch_id = f"{ch_name}ch{ch_ind}"
                    event_channels.append((ch_name, event_ch_id, "event"))
                    self._all_event_ticks[event_ch_id] = events

                elif ch_type == sp.DataType.Marker:
                    data = self.smrx_file.ReadMarkers(
                        ch_ind, int(max_time * self._time_base) * 15, 0, max_time + 1
                    )

                    times = np.array([d.Tick for d in data])
                    print(data)
                    labels = np.array([chr(int(d.Code1)) for d in data], dtype="U")

                    event_ch_id = f"{ch_name}ch{ch_ind}"
                    event_channels.append((ch_name, event_ch_id, "event"))

                    self._all_event_ticks[event_ch_id] = times
                    self._all_event_labels[event_ch_id] = labels



                elif ch_type in (sp.DataType.Adc, sp.DataType.RealWave):
                    print(f"importing {ch_type} channel")

                    first_time = self.smrx_file.FirstTime(ch_ind, 0, max_time)
                    size = int(max_time / divide)

                    chan_id = str(ch_ind)

                    signal_channels.append(
                        (
                            ch_name,          # name (kept as Neo object name)
                            chan_id,          # id (must be string)
                            sampling_rate,    # sampling_rate
                            "float32",        # dtype
                            self.smrx_file.GetChannelUnits(ch_ind) if ch_name not in ["EMG","L EMG","Virtual"] else "mV",              # units (adjust if known)
                            1.0 / self.smrx_file.GetChannelScale(ch_ind),              # gain
                            self.smrx_file.GetChannelOffset(ch_ind),              # offset
                            "0",              # stream_id (temporary)
                        )
                    )

                    channel_infos.append(
                        (
                            first_time,
                            max_time,
                            divide,
                            size,
                            sampling_rate,
                        )
                    )

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        print("signal_channels",signal_channels)

        channel_infos = np.array(
            channel_infos,
            dtype=[
                ("first_time", "i8"),
                ("max_time", "i8"),
                ("divide", "i8"),
                ("size", "i8"),
                ("sampling_rate", "f8"),
            ],
        )
        # one stream per channel (do NOT group)
        self.stream_info = channel_infos
        signal_streams = []

        for i in range(len(signal_channels)):
            stream_id = str(i)

            signal_channels["stream_id"][i] = stream_id
            stream_name = signal_channels["name"][i]

            signal_streams.append((stream_name, stream_id))

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        self._seg_t_start = np.inf
        self._seg_t_stop = -np.inf
        for info in self.stream_info:
            self._seg_t_start = min(
                self._seg_t_start, info["first_time"] * self._time_base
            )

            self._seg_t_stop = max(self._seg_t_stop, info["max_time"] * self._time_base)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

        self.raw_annotations["blocks"][0]["rec_datetime"] = self._rec_datetime
        self.raw_annotations["blocks"][0]["file_origin"] = self.filename

        self.raw_annotations["blocks"][0]["segments"][0]["rec_datetime"] = self._rec_datetime

    def _get_signal_size(self, block_index, seg_index, stream_index):
        size = self.stream_info[stream_index]["size"]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        info = self.stream_info[stream_index]
        t_start = info["first_time"] * self._time_base
        return t_start

    def _get_analogsignal_chunk(
        self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
    ):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.stream_info[stream_index]["size"]

        stream_id = self.header["signal_streams"]["id"][stream_index]
        signal_channels = self.header["signal_channels"]
        mask = signal_channels["stream_id"] == stream_id
        signal_channels = signal_channels[mask]
        if channel_indexes is not None:
            signal_channels = signal_channels[channel_indexes]

        num_chans = len(signal_channels)

        size = i_stop - i_start
        sigs = np.zeros((size, num_chans), dtype="int16")

        info = self.stream_info[stream_index]
        t_from = info["first_time"] + info["divide"] * i_start
        t_upto = info["first_time"] + info["divide"] * i_stop

        for i, ch_id in enumerate(signal_channels["id"]):
            ch_ind = int(ch_id)
            sig = self.smrx_file.ReadInts(
                chan=ch_ind, nMax=size, tFrom=t_from, tUpto=t_upto
            )
            sigs[:, i] = sig

        return sigs

    def _spike_count(self, block_index, seg_index, unit_index):
        unit_id = self.header["spike_channels"][unit_index]["id"]
        spike_ticks = self._all_spike_ticks[unit_id]
        return spike_ticks.size

    def _get_spike_timestamps(
        self, block_index, seg_index, unit_index, t_start, t_stop
    ):
        unit_id = self.header["spike_channels"][unit_index]["id"]
        spike_ticks = self._all_spike_ticks[unit_id]
        if t_start is not None:
            tick_start = int(t_start / self._time_base)
            spike_ticks = spike_ticks[spike_ticks >= tick_start]
        if t_stop is not None:
            tick_stop = int(t_stop / self._time_base)
            spike_ticks = spike_ticks[spike_ticks <= tick_stop]
        return spike_ticks

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times *= self._time_base
        return spike_times

    def _get_spike_raw_waveforms(
        self, block_index, seg_index, spike_channel_index, t_start, t_stop
    ):
        return None

    def _get_event_timestamps(
        self, block_index, seg_index, event_channel_index, t_start, t_stop
    ):
        ch_id = self.header["event_channels"][event_channel_index]["id"]

        times = self._all_event_ticks[ch_id]
        labels = self._all_event_labels.get(ch_id, None)

        return times, None, labels


    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index=0):
        event_times = event_timestamps.astype(dtype)
        event_times *= self._time_base
        return event_times

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = np.ceil(self.max_time_overall * self._time_base)
        return t_stop

    def _segment_t_start(self, block_index, seg_index):
        return 0

    def _event_count(self, block_index: int, seg_index: int, event_channel_index: int):
        return None
