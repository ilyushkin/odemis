# -*- coding: utf-8 -*-
"""
Created on 1 April 2025

@author: Patrick Cleeve

Copyright © 2025 Patrick Cleeve, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License version 2 as published by the Free
Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""

import logging
from typing import List

import wx
from odemis import gui, model
from odemis.gui import img
from odemis.acq.milling.tasks import MillingTaskSettings
from odemis.gui.comp.text import UnitFloatCtrl
from odemis.gui.comp.combo import ComboBox


class MillingPatternCheckList(wx.Panel):
    """List of pattern rows, each with an eye-icon toggle and a checkbox.

    Layout per row (mirrors the CALIBRATION panel in fastem_acq.py):

        [eye btn]  Label text                              [checkbox]

    The eye button controls whether the item is "active" (relevant to the
    current workflow selection).  The checkbox records the user's intent to
    include that pattern in a milling run.

    - When the eye is closed (item inactive / grayed): the row label is dimmed,
      the checkbox is disabled, but its checked state is preserved.
    - When the eye is re-opened: the checkbox is re-enabled with its previous
      checked state intact.
    - Checked state never changes unless the user explicitly clicks the checkbox.

    The parent sidebar is already inside a wxScrolledWindow, so this widget
    is a plain wx.Panel — its GridBagSizer reports a proper natural size to the
    FoldPanelItem so the fold panel expands correctly.
    """

    def __init__(self, parent: wx.Window):
        """
        :param parent: parent window.
        """
        super().__init__(parent, style=wx.BORDER_NONE)
        self.SetBackgroundColour(gui.BG_COLOUR_MAIN)

        self._gb = wx.GridBagSizer(vgap=2, hgap=4)
        self.SetSizer(self._gb)
        self._growable_col_set = False  # guard against double-AddGrowableCol

        # Each entry: {"name": str, "eye": ImageToggleButton,
        #              "label": wx.StaticText, "cb": wx.CheckBox, "active": bool}
        self._items: List[dict] = []
        self._chklist_handlers: List = []

    # ── Bind override ────────────────────────────────────────────────────────

    def Bind(self, event, handler=None, source=None,
             id: int = wx.ID_ANY, id2: int = wx.ID_ANY):
        """Capture EVT_CHECKLISTBOX; forward everything else to wx.

        :param event: event binder.
        :param handler: callable to invoke on checkbox clicks.
        """
        if event == wx.EVT_CHECKLISTBOX:
            if handler is not None:
                self._chklist_handlers.append(handler)
        else:
            super().Bind(event, handler, source=source, id=id, id2=id2)

    def _fire_chklist_event(self, idx: int):
        """Notify registered EVT_CHECKLISTBOX handlers with a synthetic event.

        :param idx: index of the clicked item.
        """
        class _SyntheticEvt:
            def __init__(self, i):
                self._i = i
            def GetInt(self):
                return self._i

        for handler in self._chklist_handlers:
            handler(_SyntheticEvt(idx))

    def _on_checkbox_click(self, evt: wx.CommandEvent):
        """Forward checkbox clicks to registered EVT_CHECKLISTBOX handlers.

        :param evt: wx.EVT_CHECKBOX event from a child checkbox.
        """
        cb = evt.GetEventObject()
        idx = next((i for i, item in enumerate(self._items) if item["cb"] is cb), None)
        if idx is not None:
            self._fire_chklist_event(idx)

    # ── wx.CheckListBox-compatible API ───────────────────────────────────────

    def SetItems(self, items: List[str]):
        """Replace all items.  All items start checked and active (eye open).

        :param items: list of item label strings.
        """
        # Destroy existing widgets
        for item in self._items:
            item["eye"].Destroy()
            item["label"].Destroy()
            item["cb"].Destroy()
        self._gb.Clear(False)
        self._items = []

        for row, name in enumerate(items):
            # Eye indicator (col 0) — read-only, shows active/inactive state
            eye_bmp = wx.StaticBitmap(self, bitmap=img.getBitmap("icon/ico_eye_open.png"))
            eye_bmp.SetToolTip("Indicates whether this pattern is active for the selected workflow step")

            # Label (col 0, same cell in an h_sizer)
            h_sizer = wx.BoxSizer(wx.HORIZONTAL)
            h_sizer.Add(eye_bmp, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=3)
            lbl = wx.StaticText(self, label=name)
            lbl.SetForegroundColour(gui.FG_COLOUR_MAIN)
            lbl.SetBackgroundColour(gui.BG_COLOUR_MAIN)
            h_sizer.Add(lbl, flag=wx.ALIGN_CENTER_VERTICAL)

            self._gb.Add(h_sizer, pos=(row, 0),
                         flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=2)

            # Checkbox (col 1)
            cb = wx.CheckBox(self, style=wx.ALIGN_RIGHT | wx.NO_BORDER)
            cb.SetValue(True)
            cb.SetBackgroundColour(gui.BG_COLOUR_MAIN)
            cb.Bind(wx.EVT_CHECKBOX, self._on_checkbox_click)
            self._gb.Add(cb, pos=(row, 1),
                         flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4)

            self._items.append({"name": name, "eye": eye_bmp,
                                 "label": lbl, "cb": cb, "active": True})

        if not self._growable_col_set:
            self._gb.AddGrowableCol(0, proportion=1)
            self._growable_col_set = True
        self.Layout()

    def SetCheckedStrings(self, checked: List[str]):
        """Set which items are checked by name; leaves grayed state untouched.

        :param checked: names of items to check; all others unchecked.
        """
        checked_set = set(checked)
        for item in self._items:
            item["cb"].SetValue(item["name"] in checked_set)

    def GetCheckedStrings(self) -> List[str]:
        """Return names of all checked items, including grayed-out ones.

        :return: list of checked item names.
        """
        return [item["name"] for item in self._items if item["cb"].GetValue()]

    def Check(self, idx: int, state: bool = True):
        """Check or uncheck an item by index.

        :param idx: item index.
        :param state: True to check, False to uncheck.
        """
        self._items[idx]["cb"].SetValue(state)

    def IsChecked(self, idx: int) -> bool:
        """Return whether the item at index is checked.

        :param idx: item index.
        :return: True if checked.
        """
        return self._items[idx]["cb"].GetValue()

    def GetString(self, idx: int) -> str:
        """Return the label of the item at index.

        :param idx: item index.
        :return: item label.
        """
        return self._items[idx]["name"]

    def GetCount(self) -> int:
        """Return the number of items.

        :return: item count.
        """
        return len(self._items)

    # ── Extended API ─────────────────────────────────────────────────────────

    def SetItemActive(self, idx: int, active: bool):
        """Activate or gray out an item without changing its checked state.

        Swaps the eye bitmap to open/closed, enables/disables the checkbox,
        and dims the label.  The eye indicator is not clickable — it only
        reflects the state driven by the current workflow selection.

        :param idx: item index.
        :param active: True to activate (eye open), False to gray out (eye closed).
        """
        item = self._items[idx]
        item["active"] = active
        item["eye"].SetBitmap(
            img.getBitmap("icon/ico_eye_open.png") if active
            else img.getBitmap("icon/ico_eye_closed.png")
        )
        item["cb"].Enable(active)
        item["label"].SetForegroundColour(
            gui.FG_COLOUR_MAIN if active else gui.FG_COLOUR_DIS
        )
        item["label"].Refresh()


class MillingTaskPanel(wx.Panel):
    """Panel for Milling Settings"""

    def __init__(self, parent, task: MillingTaskSettings):
        super().__init__(parent=parent, name=task.name)
        self._parent = parent
        self.SetForegroundColour(gui.FG_COLOUR_EDIT)
        self.SetBackgroundColour(gui.BG_COLOUR_MAIN)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)

        self._panel = wx.Panel(self, style=wx.TAB_TRAVERSAL | wx.NO_BORDER)
        self._panel.SetBackgroundColour(gui.BG_COLOUR_MAIN)
        self._panel.SetForegroundColour(gui.FG_COLOUR_MAIN)
        self._panel.SetFont(self.GetFont())

        self.gb_sizer = wx.GridBagSizer()
        self._panel.SetSizer(self.gb_sizer)

        self.main_sizer.Add(self._panel, 1, wx.ALL | wx.EXPAND, 5)

        self.num_rows = 0
        self.task = task

        # header
        title = self._add_side_label(task.name)
        font = title.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        font.SetPointSize(font.GetPointSize() + 1)
        title.SetFont(font)
        self.num_rows += 1

        # map of control fields
        self.ctrl_dict = {}

        CONFIG = {
            "current": {"label": "Current", "accuracy": 2, "unit": "A"},
            "align": {"label": "Align at milling current"},
            "mode": {"label": "Milling mode"},
            "width": {"label": "Width", "accuracy": 2, "unit": "m"},
            "height": {"label": "Height", "accuracy": 2, "unit": "m"},
            "depth": {"label": "Depth", "accuracy": 2, "unit": "m"},
            "spacing": {"label": "Spacing", "accuracy": 2, "unit": "m"},
            "width_top": {"label": "Width (Top)", "accuracy": 2, "unit": "m"},
            "height_top": {"label": "Height (Top)", "accuracy": 2, "unit": "m"},
            "width_bottom": {"label": "Width (Bottom)", "accuracy": 2, "unit": "m"},
            "height_bottom": {"label": "Height (Bottom)", "accuracy": 2, "unit": "m"},
        }

        unsupported_parameters = ["name", "rotation",
                                  "center", "channel",
                                  "field_of_view", "voltage",
                                  "rate", "dwell_time",
                                  "scan_direction"]

        for param in vars(task.milling):
            if param in unsupported_parameters:
                continue

            conf = CONFIG.get(param, {})
            label = conf.get("label", param)
            conf.pop("label", None)

            val = getattr(task.milling, param)
            self._add_value_field(label, val, conf, param=param)

        pattern = task.patterns[0]

        for i, pattern in enumerate(task.patterns):
            if len(task.patterns) > 1:
                # sub-header label for each pattern when there are multiple
                sub_label = self._add_side_label(pattern.name.value)
                font = sub_label.GetFont()
                font.SetWeight(wx.FONTWEIGHT_BOLD)
                sub_label.SetFont(font)
                self.num_rows += 1

            for param in vars(pattern):

                if param in unsupported_parameters:
                    continue

                conf = CONFIG.get(param, {})
                label = conf.get("label", param)
                conf.pop("label", None)

                val = getattr(pattern, param)
                self._add_value_field(label, val, conf, param=f"{i}_{param}")

        # Fit sizer
        self.main_sizer.AddSpacer(5)
        self.SetSizerAndFit(self.main_sizer)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Layout()
        self._parent.Refresh()

    def _add_value_field(self, label, val, conf, param: str):
        """Add a value field to the panel (label, ctrl)"""
        lbl_ctrl = self._add_side_label(label)
        value_ctrl = self._add_value_ctrl(val, conf)

        if value_ctrl is None:
            logging.debug(f"Unsupported parameter: {param}, {val}")
            return

        self.ctrl_dict[param] = value_ctrl
        self.gb_sizer.Add(value_ctrl, (self.num_rows, 1),
                        flag=wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL,
                        border=5)
        # row height for milling pattern propeties controls
        row_height = 18
        # column width for milling pattern properties controls
        min_col_width = 120
        self.gb_sizer.SetItemMinSize(value_ctrl, min_col_width, row_height)
        self.gb_sizer.SetItemMinSize(lbl_ctrl, min_col_width, row_height)

        value_ctrl.SetForegroundColour(gui.FG_COLOUR_EDIT)
        value_ctrl.SetBackgroundColour(gui.BG_COLOUR_MAIN)
        self.num_rows += 1

    def _add_value_ctrl(self, val, conf):
        """Add a control for a value"""
        value_ctrl = None
        if isinstance(val, model.StringEnumerated):
            value_ctrl = ComboBox(self._panel, value=val.value,
                        choices=val.choices, style=wx.CB_READONLY | wx.BORDER_NONE)
        if isinstance(val, model.FloatContinuous):
            value_ctrl = UnitFloatCtrl(self._panel, value=val.value,
                                        style=wx.NO_BORDER, **conf)
        if isinstance(val, model.BooleanVA):
            value_ctrl = wx.CheckBox(self._panel, **conf)
            value_ctrl.SetValue(val.value)

        return value_ctrl

    def _add_side_label(self, label_text, tooltip=None):
        """ Add a text label to the control grid

        This method should only be called from other methods that add control to the control grid

        :param label_text: (str)
        :return: (wx.StaticText)

        """

        lbl_ctrl = wx.StaticText(self._panel, -1, label_text)
        if tooltip:
            lbl_ctrl.SetToolTip(tooltip)

        self.gb_sizer.Add(lbl_ctrl, (self.num_rows, 0),
                        flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        return lbl_ctrl

        # ref: add_setting_entry

    def _on_size(self, event):
        """ Handle the wx.EVT_SIZE event for the Expander class """
        self.SetSize((self._parent.GetSize().x, -1))
        self.Layout()
        self.Refresh()
        event.Skip()

    def collapse(self, collapse):
        """ Collapses or expands the pane window """

        if self._collapsed == collapse:
            return

        self.Freeze()

        # update our state
        self._panel.Show(not collapse)
        self._collapsed = collapse

        # Call after is used, so the fit will occur after everything has been hidden or shown
        # wx.CallAfter(self.Parent.fit_streams)

        self.Thaw()

    # GUI events: update the stream when the user changes the values

    def on_visibility_btn(self, evt):
        # generate EVT_STREAM_VISIBLE
        return
