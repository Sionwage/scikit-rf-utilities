import numpy as np
import skrf
from skrf.media import DefinedGammaZ0

calKitDefinitions = {
    "Keysight 85032F": {
        "male": {
            "C_0": 89.939 * 10**-15,  # F, Male Calibration Open
            "C_1": 2536.800 * 10**-27,  # F/Hz
            "C_2": -264.990 * 10**-36,  # F/Hz^2
            "C_3": 13.400 * 10**-45,  # F/Hz^3
            "L_0": 3.3998 * 10**-12,  # H, Male Short
            "L_1": -496.4808 * 10**-24,  # H/Hz
            "L_2": 34.8314 * 10**-33,  # H/Hz^2
            "L_3": -0.7847 * 10**-42,  # H/Hz^3
            "R": 50,  # ohms
            "OffsetZ_0_Open": 50,  # ohms
            "OffsetZ_0_Short": 49.992,  # ohms
            "OffsetZ_0_Load": 50,  # ohms
            "OffsetDelay_Open": 4.0856 * 10**-11,  # Sec
            "OffsetDelay_Short": 4.5955 * 10**-11,  # Sec
            "OffsetDelay_Load": 0.0,  # Sec
            "OffsetLoss_Open": 0.93,  # Gohm/Sec
            "OffsetLoss_Short": 1.087,  # Gohm/Sec
            "OffsetLoss_Load": 0,  # Gohm/Sec
            "Standards": {"Open": 8, "Short": 7, "Load": 3},
        },
        "female": {
            "C_0": 89.939 * 10**-15,  # F, Male Calibration Open
            "C_1": 2536.800 * 10**-27,  # F/Hz
            "C_2": -264.990 * 10**-36,  # F/Hz^2
            "C_3": 13.400 * 10**-45,  # F/Hz^3
            "L_0": 3.3998 * 10**-12,  # H, Male Short
            "L_1": -496.4808 * 10**-24,  # H/Hz
            "L_2": 34.8314 * 10**-33,  # H/Hz^2
            "L_3": -0.7847 * 10**-42,  # H/Hz^3
            "R": 50,  # ohms
            "OffsetZ_0_Open": 50,  # ohms
            "OffsetZ_0_Short": 49.99,  # ohms
            "OffsetZ_0_Load": 50,  # ohms
            "OffsetDelay_Open": 4.1170 * 10**-11,  # Sec
            "OffsetDelay_Short": 4.5955 * 10**-11,  # Sec
            "OffsetDelay_Load": 0.0,  # Sec
            "OffsetLoss_Open": 0.93,  # Gohm/Sec
            "OffsetLoss_Short": 1.087,  # Gohm/Sec
            "OffsetLoss_Load": 0,  # Gohm/Sec
            "Standards": {"Open": 2, "Short": 1, "Load": 6},
        },
        "Thru": {"Standards": {"Thru": 4}},
    }
}


def create_ideal_cal_response(
    freq=skrf.Frequency(0.03, 6000, 1601, "MHz"),
    calkit=None,
    gender="male",
):
    if calkit is None:
        raise Exception("No Calibration Kit terms provided!")
    calKitTerms = calkit[gender.lower()]

    C_0 = calKitTerms["C_0"]
    C_1 = calKitTerms["C_1"]
    C_2 = calKitTerms["C_2"]
    C_3 = calKitTerms["C_3"]
    # Male Short
    L_0 = calKitTerms["L_0"]
    L_1 = calKitTerms["L_1"]
    L_2 = calKitTerms["L_2"]
    L_3 = calKitTerms["L_3"]

    # R = calKitTerms["R"]

    OffsetZ_0_Open = calKitTerms["OffsetZ_0_Open"]
    OffsetZ_0_Short = calKitTerms["OffsetZ_0_Short"]
    OffsetZ_0_Load = calKitTerms["OffsetZ_0_Load"]

    OffsetDelay_Open = calKitTerms["OffsetDelay_Open"]
    OffsetDelay_Short = calKitTerms["OffsetDelay_Short"]
    OffsetDelay_Load = calKitTerms["OffsetDelay_Load"]

    OffsetLoss_Open = calKitTerms["OffsetLoss_Open"]
    OffsetLoss_Short = calKitTerms["OffsetLoss_Short"]
    OffsetLoss_Load = calKitTerms["OffsetLoss_Load"]

    def keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0):
        if offset_delay or offset_loss:
            alpha_l = (offset_loss * offset_delay) / (2 * offset_z0)
            alpha_l *= np.sqrt(freq.f / 1e9)
            beta_l = 2 * np.pi * freq.f * offset_delay + alpha_l
            zc = offset_z0 + (1 - 1j) * (offset_loss / (4 * np.pi * freq.f)) * np.sqrt(
                freq.f / 1e9
            )
            gamma_l = alpha_l + beta_l * 1j

            medium = DefinedGammaZ0(frequency=freq, z0=zc, gamma=gamma_l)
            offset_line = medium.line(d=1, unit="m")
            return medium, offset_line
        else:
            medium = DefinedGammaZ0(frequency=freq, z0=offset_z0)
            line = medium.line(d=0)
            return medium, line

    def keysight_calkit_open(
        freq, offset_delay, offset_loss, c0, c1, c2, c3, offset_z0=50
    ):
        medium, line = keysight_calkit_offset_line(
            freq, offset_delay, offset_loss, offset_z0
        )
        # Capacitance is defined with respect to the port impedance offset_z0, not the lossy
        # line impedance. In scikit-rf, the return values of `shunt_capacitor()` and `medium.open()`
        # methods are (correctly) referenced to the port impedance.
        if c0 or c1 or c2 or c3:
            poly = np.poly1d([c3, c2, c1, c0])
            capacitance = medium.shunt_capacitor(poly(freq.f)) ** medium.open()
        else:
            capacitance = medium.open()
        return line**capacitance

    def keysight_calkit_short(
        freq, offset_delay, offset_loss, l0, l1, l2, l3, offset_z0=50
    ):
        # Inductance is defined with respect to the port impedance offset_z0, not the lossy
        # line impedance. In scikit-rf, the return values of `inductor()` and `medium.short()`
        # methods are (correctly) referenced to the port impedance.
        medium, line = keysight_calkit_offset_line(
            freq, offset_delay, offset_loss, offset_z0
        )
        if l0 or l1 or l2 or l3:
            poly = np.poly1d([l3, l2, l1, l0])
            inductance = medium.inductor(poly(freq.f)) ** medium.short()
        else:
            inductance = medium.short()
        return line**inductance

    def keysight_calkit_load(freq, offset_delay=0, offset_loss=0, offset_z0=50):
        medium, line = keysight_calkit_offset_line(
            freq, offset_delay, offset_loss, offset_z0
        )
        load = medium.match()
        return line**load

    def keysight_calkit_thru(freq, offset_delay=0, offset_loss=0, offset_z0=50):
        medium, line = keysight_calkit_offset_line(
            freq, offset_delay, offset_loss, offset_z0
        )
        thru = medium.thru()
        return line**thru

    open_std = keysight_calkit_open(
        freq,
        offset_delay=OffsetDelay_Open,
        offset_loss=OffsetLoss_Open,
        c0=C_0,
        c1=C_1,
        c2=C_2,
        c3=C_3,
        offset_z0=OffsetZ_0_Open,
    )
    short_std = keysight_calkit_short(
        freq,
        offset_delay=OffsetDelay_Short,
        offset_loss=OffsetLoss_Short,
        l0=L_0,
        l1=L_1,
        l2=L_2,
        l3=L_3,
        offset_z0=OffsetZ_0_Short,
    )
    load_std = keysight_calkit_load(
        freq,
        offset_delay=OffsetDelay_Load,
        offset_loss=OffsetLoss_Load,
        offset_z0=OffsetZ_0_Load,
    )
    thru_std = keysight_calkit_thru(freq)

    for ntwk in [short_std, open_std, load_std, thru_std]:
        ntwk.renormalize(50)

    return [short_std, open_std, load_std, thru_std]
