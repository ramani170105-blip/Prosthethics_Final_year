from __future__ import annotations

import mne
import numpy as np


class EEGEventExtractor:

    def __init__(self, raw):

        self.raw = raw

        self.events = None

        self.event_id = None

        self.labels = None

    # =====================================
    # Extract Events
    # =====================================

    def extract_events(self):

        events, event_id = mne.events_from_annotations(
            self.raw
        )

        self.events = events
        self.event_id = event_id

        print("\n" + "="*60)
        print("EVENT EXTRACTION")
        print("="*60)

        print(f"Events Found : {len(events)}")

        print("\nEvent IDs")

        for name, value in event_id.items():

            print(f"{name} : {value}")

        self.decode_labels()
        return events, event_id

    # =====================================
    # Decode Labels
    # =====================================

    def decode_labels(self):

        if self.events is None:

            self.extract_events()

        reverse = {

            value: key

            for key, value in self.event_id.items()

        }

        labels = [

            reverse[event[2]]

            for event in self.events

        ]

        self.labels = labels

        return labels

    # =====================================
    # Summary
    # =====================================

    def summary(self):

        print("\n" + "="*60)
        print("EVENT SUMMARY")
        print("="*60)

        print(f"Total Events : {len(self.events)}")

        unique = np.unique(self.labels)

        print("\nLabels")

        for label in unique:

            print(label)


    def get_events(self):

        return {

        "events": self.events,

        "event_id": self.event_id,

        "labels": self.labels

        }