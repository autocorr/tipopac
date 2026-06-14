from taskinit import *
from casac import casac
import numpy as np
import scipy.optimize

# tipopac is released under a BSD 3-Clause License
# See LICENSE for details

# HISTORY:
#   1.0  22Oct2019  Initial version.
#


def tipopac(
    msname, caltableZ, tauPerAnt, calcTcals, caltableT, cmdFlag, usrFlag, flagFile
):

    #
    # Task tipopac
    #
    #    Derive zenith opacity and Tcals from JVLA tip data.
    #    Christopher A. Hales
    #
    #    Based on JIRA CASR-16, VLA Sci. Memo 170, and EVLA Memos 145 and 202.
    #    Originally written to support ngVLA Memo 63.
    #    See xml for code overview.
    #
    #    Version 1.0 (tested with CASA Version 5.6.0 REL)
    #    22 October 2019

    casalog.origin("tipopac")
    casalog.post("--> tipopac version 1.0")

    # JVLA tips runs between approx 55-23 deg elevation (35-67 deg zenith angle, respectively)
    porder = 3  # order of polynomial fit
    zmin = 40.0  # zenith angle min, degrees
    zmax = 62.0  # zenith angle max, degrees

    # minimum number of (assuming 1 sec) integration times to get a good solution
    # pick ~40 sec somewhat randomly.  A normal tipping scan should take ~90 sec.
    minTipInts = 40

    # for calcTcals=True, set threshold warning level for % diff with TcalMS
    Tdifthresh = 30

    if calcTcals & tauPerAnt:
        casalog.post(
            "*** WARNING: Setting tauPerAnt=False because calcTcals=True.", "WARN"
        )
        tauPerAnt = False

    # avoid using global cb tool, which can cause issues with table cache
    def gencaltableZ(msname, caltableZ):
        mycb = casac.calibrater()
        mycb.open(msname, False, False, False)
        mycb.createcaltable(caltableZ, "Real", "TOpac", True)
        mycb.close()

    # temperature of resolved astrophysical background in noise K
    def get_Trab(nu_Hz):
        # Condon et al., 2012, ApJ, 758, 23
        return 0.1 / (nu_Hz / 1.4e9) ** (2.7)

    # kinetic to noise temp in K
    def k2nt(T, nu_Hz):
        h = 6.6261e-34
        k = 1.3806e-23
        return T * (h * nu_Hz / (k * T) / (np.exp(h * nu_Hz / (k * T)) - 1.0))

    # temperature of unresolved astrophysical background in noise K
    def get_Tuab(nu_Hz):
        Tcmb = 2.725
        return k2nt(Tcmb, nu_Hz)

    # Tsys vs z tipping curve function
    def func(z, params, Trab, Tuab, Twmtp):
        # z in deg, T's all in noise K
        # T0 = Tant + Trx1 + Trx2 + Tcal/2 ~= constant
        T0, tau0 = params
        Tsys = (
            T0
            + (Trab + Tuab) * np.exp(-tau0 / np.cos(np.deg2rad(z)))
            + Twmtp * (1 - np.exp(-tau0 / np.cos(np.deg2rad(z))))
        )
        return Tsys

    # option 1: calcTcals=False and tauPerAnt=True
    #           3 unknown parameters: T0_pol0, T0_pol1, tau0
    # option 2: calcTcals=False and tauPerAnt=False
    #           2*nant+1 unknown parameters:
    #             T0_a0_p0, T0_a0_p1, ..., T0_aN_p0, T0_aN_p1, tau0
    # option 3: same as 2, but Tsys values will be adjusted beforehand
    # these can share the same wrapper (op1: N=1, op2/3: N=nant)
    def err_multi_wrap(Trab, Tuab, Twmtp):
        def err_multi(p, *argv):
            N = len(argv) / 3
            z = argv[:N]
            Tsys = argv[N:]
            params = p[0], p[-1]
            errArr = Tsys[0] - func(z[0], params, Trab, Tuab, Twmtp)
            for k in range(1, 2 * N):
                params = p[k], p[-1]
                errArr = np.concatenate(
                    [errArr, Tsys[k] - func(z[k / 2], params, Trab, Tuab, Twmtp)]
                )

            return errArr

        return err_multi

    ### get antenna details
    tb.open(msname + "/ANTENNA")
    antNames = tb.getcol("NAME")
    tb.close()
    lenAnt = len(antNames)

    ### get full spw details for MS (not only tipping scans)
    tb.open(msname + "/SPECTRAL_WINDOW")
    spwCntFreq = np.mean(tb.getcol("CHAN_FREQ"), axis=0)
    tb.close()
    lenSpw = len(spwCntFreq)

    ### get pointing zenith angle
    # check if pointing sub-table contains data. If not, give error and exit.
    # this should be produced by importasdm in all cases (with_pointing_correction true or false)
    casalog.post("--> Reading antenna pointing data.")
    tb.open(msname + "/POINTING")

    # read in elevation vs time, will include data from tips and also pointing if performed
    # Note this is stored in ENCODER in AZELGEO coordinates
    # tb.getcolkeywords('ENCODER')
    # {'MEASINFO': {'Ref': 'AZELGEO', 'type': 'direction'},
    # 'QuantumUnits': array(['rad', 'rad'], dtype='|S4')}
    #
    # CASA coordinate frames:
    # https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/coordinate-frames

    # extract pointing data for full observation
    # not super efficient, but majority of data is expected to come from tipping scans.
    # first, get max timestamps per antenna, in case they differ
    maxAntT = 0
    for a in range(lenAnt):
        temptb = tb.query("ANTENNA_ID==" + str(a))
        lenAntT = len(temptb.getcol("TIME"))
        if lenAntT > maxAntT:
            maxAntT = lenAntT

    if maxAntT == 0:
        casalog.post("*** ERROR: That's strange, the pointing table is empty.", "ERROR")
        casalog.post("*** ERROR: Exiting tipopac.", "ERROR")
        return

    # 0 = time UTC seconds, 1 = zenith angle (deg)
    dataPoint = np.zeros([lenAnt, maxAntT, 2])
    me.doframe(me.observatory("VLA"))
    for a in range(lenAnt):
        casalog.post(
            "    processing antenna "
            + antNames[a]
            + " ("
            + str(a + 1)
            + "/"
            + str(lenAnt)
            + ")"
        )
        temptb = tb.query("ANTENNA_ID==" + str(a))
        lenAntT = len(temptb.getcol("TIME"))
        dataPoint[a, 0:lenAntT, 0] = temptb.getcol("TIME")
        azel = temptb.getcol("ENCODER")
        for i in range(lenAntT):
            dataPoint[a, i, 1] = 90 - np.rad2deg(
                me.measure(
                    me.direction(
                        "AZELGEO",
                        str(np.rad2deg(azel[0, i])) + "deg",
                        str(np.rad2deg(azel[1, i])) + "deg",
                    ),
                    "AZEL",
                )["m1"]["value"]
            )

    tb.close()
    del temptb, azel

    ### read in time ranges for tipping scans and get associated spw's
    casalog.post("--> Reading time ranges for tipping scans.")
    msmd = casac.msmetadata()
    msmd.open(msname)
    # Only the scan with 2 subscans, with intent DO_SKYDIP, are of interest.
    # The others can be ignored, they don't contain any useful data.
    scans = msmd.scansforintent("*DO_SKYDIP*")
    lenScans = len(scans)
    tipSpw = msmd.spwsforintent("*DO_SKYDIP*")
    lenTipSpw = len(tipSpw)
    # get start and end time for each scan
    times = np.zeros([lenScans, 2])  # UTC seconds
    for i in range(lenScans):
        times[i, 0] = msmd.timesforscan(scans[i])[0]
        times[i, 1] = msmd.timesforscan(scans[i])[-1]

    msmd.done()

    ### get estimated weighted mean atmospheric temperatures in kinetic temp K
    casalog.post(
        "--> Reading MS surface temperature data and "
        + "estimating weighted mean atmospheric temperatures."
    )
    # sampled every approximately 1 minute
    tb.open(msname + "/WEATHER")
    tmp1 = tb.getcol("TIME")
    # estimate weighted mean atmospheric temperature using
    # Tm ~ 70.2 + 0.72 * Ts
    # from Bevis et al., 1992, J. Geophys. Res. 97(D14), 15,787
    tmp2 = tb.getcol("TEMPERATURE") * 0.72 + 70.2
    tmp2[tb.getcol("TEMPERATURE_FLAG") == 1] = np.NaN
    tb.close()
    # time (UTC sec), temp (K)
    dataTemp = np.column_stack((tmp1, tmp2))
    del tmp1, tmp2

    ### read in online flags except for ANTENNA_NOT_ON_SOURCE
    # subreflector errors shouldn't make any difference, but no harm flagging
    if cmdFlag:
        casalog.post("--> Reading online flags.")
        tb.open(msname + "/FLAG_CMD")
        if len(tb.getcol("REASON")) == 0:
            casalog.post(
                "*** ERROR: The online flag table (FLAG_CMD) is empty.", "ERROR"
            )
            casalog.post(
                "*** ERROR: Ensure that process_flags=True when running "
                + "importasdm.",
                "ERROR",
            )
            casalog.post("*** ERROR: Exiting tipopac.", "ERROR")
            return

        temptb = tb.query("REASON!='ANTENNA_NOT_ON_SOURCE'")
        dataCmdRaw = temptb.getcol("COMMAND")
        lenDataCmd = len(dataCmdRaw)
        # antenna, start time, end time (UTC sec)
        dataCmd = np.zeros([lenDataCmd, 3])
        for f in range(lenDataCmd):
            dataCmd[f, 0] = np.where(
                antNames == dataCmdRaw[f].replace("'", "&&").split("&&")[1]
            )[0][0]
            dataCmd[f, 1] = (
                qa.quantity(
                    dataCmdRaw[f].replace("'", "&&").split("&&")[4].split("~")[0], "ymd"
                )["value"]
                * 24
                * 3600
            )
            dataCmd[f, 2] = (
                qa.quantity(
                    dataCmdRaw[f].replace("'", "&&").split("&&")[4].split("~")[1], "ymd"
                )["value"]
                * 24
                * 3600
            )

        tb.close()

    ### read in user-specified flags
    if usrFlag:
        casalog.post("--> Reading user-defined flags.")
        dataUsrRaw = np.loadtxt(flagFile, dtype=str)
        if dataUsrRaw.size == 3:
            dataUsrRaw = dataUsrRaw.reshape([1, 3])

        lenDataUsr = len(dataUsrRaw)
        # antenna, spw, start time, end time (UTC sec)
        dataUsr = np.zeros([lenDataUsr, 4])
        mintime = times.min()
        maxtime = times.max()
        maxf = lenDataUsr
        f = 0
        for fraw in range(lenDataUsr):
            allant = False
            allspw = False
            myant = dataUsrRaw[fraw, 0].replace("'", "").split("=")[1]
            myspw = dataUsrRaw[fraw, 1].replace("'", "").split("=")[1]
            mytime1 = (
                qa.quantity(
                    dataUsrRaw[fraw, 2].replace("'", "").split("=")[1].split("~")[0],
                    "ymd",
                )["value"]
                * 24
                * 3600
            )
            mytime2 = (
                qa.quantity(
                    dataUsrRaw[fraw, 2].replace("'", "").split("=")[1].split("~")[1],
                    "ymd",
                )["value"]
                * 24
                * 3600
            )
            if myant == "-1":
                allant = True
            else:
                dataUsr[f, 0] = np.where(antNames == myant)[0][0]

            if myspw == "-1":
                allspw = True
            else:
                dataUsr[f, 1] = float(myspw)

            dataUsr[f, 2] = mytime1
            dataUsr[f, 3] = mytime2

            # to avoid having to modify code below, copy flag command to all
            # relevant ants, spws, times (not the most elegant solution...)
            # only focus on following cases, within given time range:
            # ant and spw specified
            # ant specified with allspw
            # allant and allspw
            if (not allant) and (not allspw):
                f += 1
            elif (not allant) and (allspw):
                # all spws for a given antenna in a given time range
                dataUsr = np.insert(dataUsr, f, np.zeros([lenSpw - 1, 4]), axis=0)
                for s in range(lenSpw):
                    dataUsr[f, 0] = np.where(antNames == myant)[0][0]
                    dataUsr[f, 1] = s
                    dataUsr[f, 2] = mytime1
                    dataUsr[f, 3] = mytime2
                    f += 1

                maxf += lenSpw - 1
            elif (allant) and (allspw):
                # all antennas and all spws in a given time range
                dataUsr = np.insert(
                    dataUsr, f, np.zeros([lenAnt * lenSpw - 1, 4]), axis=0
                )
                for a in range(lenAnt):
                    for s in range(lenSpw):
                        dataUsr[f, 0] = a
                        dataUsr[f, 1] = s
                        dataUsr[f, 2] = mytime1
                        dataUsr[f, 3] = mytime2
                        f += 1

                maxf += lenAnt * lenSpw - 1
            else:
                casalog.post(
                    "*** ERROR: Manual flags not specified "
                    + "according to instructions in help.",
                    "ERROR",
                )
                casalog.post("*** ERROR: Exiting tipopac.", "ERROR")
                return

    ### read in switched power psum and pdif per tip scan and apply flags
    casalog.post("--> Reading switched power data.")
    # start by reading in stored Tcals (K)
    tb.open(msname + "/CALDEVICE")
    # antenna, spw, pol
    dataTcalMS = np.zeros([lenAnt, lenSpw, 2])
    for a in range(lenAnt):
        for s in range(lenSpw):
            temptb = tb.query(
                "ANTENNA_ID==" + str(a) + "&&SPECTRAL_WINDOW_ID==" + str(s)
            )
            # rows 0/1 = noise tube/solar filter
            # cols 0/1 = R/L
            dataTcalMS[a, s, 0] = temptb.getcol("NOISE_CAL")[0, 0, 0]
            dataTcalMS[a, s, 1] = temptb.getcol("NOISE_CAL")[0, 1, 0]

    tb.close()

    # create Z caltable to be filled in below
    gencaltableZ(msname, caltableZ)
    # set default values with everything flagged
    caltableNrows = lenScans * lenSpw * lenAnt
    tb.open(caltableZ, nomodify=False)
    tb.addrows(caltableNrows)
    k = 0
    for i in range(lenScans):
        for a in range(lenAnt):
            for s in range(lenSpw):
                tb.putcell("TIME", k, (times[i, 0] + times[i, 1]) / 2.0)
                tb.putcell("FIELD_ID", k, -1)
                tb.putcell("SPECTRAL_WINDOW_ID", k, s)
                tb.putcell("ANTENNA1", k, a)
                tb.putcell("ANTENNA2", k, -1)
                tb.putcell("SCAN_NUMBER", k, i)
                tb.putcell("FPARAM", k, np.array([[0.0]]))
                tb.putcell("PARAMERR", k, np.array([[0.0]]))
                tb.putcell("FLAG", k, np.array([[True]], dtype=bool))
                tb.putcell("SNR", k, np.array([[1.0]]))
                # the WEIGHT column can be left with empty cells
                k += 1

    tb.flush()
    tb.close()
    if calcTcals:
        tb.open(msname + "/CALDEVICE")
        newtab = tb.copy(
            caltableT, deep=True, valuecopy=True, norows=True, returnobject=True
        )
        tb.close()
        newtab.close()
        tb.open(caltableT, nomodify=False)
        tb.addrows(caltableNrows)
        k = 0
        for i in range(lenScans):
            for a in range(lenAnt):
                for s in range(lenSpw):
                    tb.putcell("ANTENNA_ID", k, a)
                    tb.putcell("SPECTRAL_WINDOW_ID", k, s)
                    tb.putcell("TIME", k, (times[i, 0] + times[i, 1]) / 2.0)
                    tb.putcell("NUM_CAL_LOAD", k, 2)
                    tb.putcell(
                        "CAL_LOAD_NAMES",
                        k,
                        np.array([["NOISE_TUBE_LOAD"], ["SOLAR_FILTER"]]),
                    )
                    tb.putcell("NUM_RECEPTOR", k, 2)
                    tb.putcell("NOISE_CAL", k, np.array([[0.0, 0.0], [0.0, 0.0]]))
                    # other columns can default
                    k += 1

        tb.flush()
        tb.close()

    # first, get all data wrt zenith angle into a master array
    # JVLA tipping scans don't run for more than 2 mins with 1 sec sampling
    # scan, ant, spw, pol, timestamp: 0=ZA[deg], 1=Twmt[kinetic K], 2=Tsys'[K]
    #                   if calcTcals: 3=delta(Tsys') between z_min and z_max from polynomial fit,
    # this isn't very efficient; meh, the runtime is dominated by getcol, and data1 size isn't huge
    # (ZA and Twmt are not spw or pol dependent ... indeed Twmt isn't even antenna dependent, meh2)
    # (and Twmt will be effectively time-independent during a scan, meh3)
    if calcTcals:
        data1 = np.zeros([lenScans, lenAnt, lenSpw, 2, 120, 3 + 1])
    else:
        data1 = np.zeros([lenScans, lenAnt, lenSpw, 2, 120, 3])

    tb.open(msname + "/SYSPOWER")
    for i in range(lenScans):
        casalog.post(
            "--> Gathering data for scan "
            + str(scans[i])
            + " ("
            + str(i + 1)
            + "/"
            + str(lenScans)
            + ")"
        )
        casalog.filter("WARN")
        msmd.open(msname)
        scanspws = msmd.spwsforscan(scans[i])
        lenScanSpws = len(scanspws)
        msmd.done()
        casalog.filter("INFO")
        for a in range(lenAnt):
            casalog.post(
                "    antenna "
                + antNames[a]
                + " ("
                + str(a + 1)
                + "/"
                + str(lenAnt)
                + ")"
            )
            for s in scanspws:
                casalog.post(
                    "      spw "
                    + str(s)
                    + " ("
                    + str(s - scanspws[0] + 1)
                    + "/"
                    + str(lenScanSpws)
                    + ")"
                )
                subtb = tb.query(
                    "TIME>="
                    + str(times[i, 0])
                    + "&&TIME<="
                    + str(times[i, 1])
                    + "&&ANTENNA_ID=="
                    + str(a)
                    + "&&SPECTRAL_WINDOW_ID=="
                    + str(s)
                )
                spT = subtb.getcol("TIME")
                lenSpT = len(spT)
                if lenSpT > 0:
                    # cols 0/1 = R/L
                    pdif = subtb.getcol("SWITCHED_DIFF")
                    psum = subtb.getcol("SWITCHED_SUM")
                    # rq   = subtb.getcol('REQUANTIZER_GAIN')
                    # rq not needed.  Pdif reported in MS appears to neglect digital gain factor

                    # apply online flags except for ANTENNA_NOT_ON_SOURCE
                    # there is probably a smarter way to do this...
                    # cmd case 1: flagging starts and ends within tip
                    tmp = dataCmd[
                        np.where(
                            (dataCmd[:, 0] == a)
                            & (dataCmd[:, 1] >= spT[0])
                            & (dataCmd[:, 2] <= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )

                    # cmd case 2: flagging starts before tip and ends within tip
                    tmp = dataCmd[
                        np.where(
                            (dataCmd[:, 0] == a)
                            & (dataCmd[:, 1] <= spT[0])
                            & (dataCmd[:, 2] >= spT[0])
                            & (dataCmd[:, 2] <= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )

                    # cmd case 3: flagging starts during tip and ends after tip
                    tmp = dataCmd[
                        np.where(
                            (dataCmd[:, 0] == a)
                            & (dataCmd[:, 1] >= spT[0])
                            & (dataCmd[:, 1] <= spT[-1])
                            & (dataCmd[:, 2] >= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )

                    # cmd case 4: flagging starts before tip and ends after tip
                    tmp = dataCmd[
                        np.where(
                            (dataCmd[:, 0] == a)
                            & (dataCmd[:, 1] <= spT[0])
                            & (dataCmd[:, 2] >= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 1]) & (spT <= tmp[f, 2]))] = (
                            np.NaN
                        )

                    # apply manual flags
                    # usr case 1: flagging starts and ends within tip
                    tmp = dataUsr[
                        np.where(
                            (dataUsr[:, 0] == a)
                            & (dataUsr[:, 1] == s)
                            & (dataUsr[:, 2] >= spT[0])
                            & (dataUsr[:, 3] <= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )

                    # usr case 2: flagging starts before tip and ends within tip
                    tmp = dataUsr[
                        np.where(
                            (dataUsr[:, 0] == a)
                            & (dataUsr[:, 1] == s)
                            & (dataUsr[:, 2] <= spT[0])
                            & (dataUsr[:, 3] >= spT[0])
                            & (dataUsr[:, 3] <= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )

                    # usr case 3: flagging starts during tip and ends after tip
                    tmp = dataUsr[
                        np.where(
                            (dataUsr[:, 0] == a)
                            & (dataUsr[:, 1] == s)
                            & (dataUsr[:, 2] >= spT[0])
                            & (dataUsr[:, 2] <= spT[-1])
                            & (dataUsr[:, 3] >= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )

                    # usr case 4: flagging starts before tip and ends after tip
                    tmp = dataUsr[
                        np.where(
                            (dataUsr[:, 0] == a)
                            & (dataUsr[:, 1] == s)
                            & (dataUsr[:, 2] <= spT[0])
                            & (dataUsr[:, 3] >= spT[-1])
                        )
                    ]
                    for f in range(len(tmp)):
                        pdif[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )
                        psum[:, np.where((spT >= tmp[f, 2]) & (spT <= tmp[f, 3]))] = (
                            np.NaN
                        )

                    del tmp

                    # put the following info into data1 (for each poln, meh)
                    for x in range(lenSpT):
                        # get pointing zenith angle in deg at spT times
                        # pointing data for the JVLA is recorded every approximately 0.1 sec
                        # don't bother interpolating to swpow timestamps, which for
                        # the JVLA are recorded every 1 sec.  Just take nearest value.
                        data1[i, a, s, :, x, 0] = dataPoint[
                            a, (np.abs(dataPoint[a, :, 0] - spT[x])).argmin(), 1
                        ]

                        # get Tatm in kinetic K at spT times
                        # MS Tsurf is only sampled approximately every minute
                        # So it's perhaps worth interpolating temperatures at the switched power timestamps
                        data1[i, a, s, :, x, 1] = np.interp(
                            spT[x], dataTemp[:, 0], dataTemp[:, 1]
                        )

                    for p in range(2):
                        # calculate Tsys = (Psum/2)/Pdif * Tcal
                        data1[i, a, s, p, 0:lenSpT, 2] = (
                            (psum[p] / 2.0) / (pdif[p]) * dataTcalMS[a, s, p]
                        )

                        # flag Tsys if pdif<0 or psum<0
                        tmpIndx = np.where(pdif[p] < 0)[0]
                        data1[i, a, s, p, tmpIndx, 2] = np.NaN
                        tmpIndx = np.where(psum[p] < 0)[0]
                        data1[i, a, s, p, tmpIndx, 2] = np.NaN

                        if len(np.where(data1[i, a, s, p, :, 2] > 0)[0]) < minTipInts:
                            data1[i, a, s, p, :, 2] = np.NaN
                            if p == 0:
                                polstr = "R"
                            else:
                                polstr = "L"

                            casalog.post(
                                "*** WARNING: "
                                + antNames[a]
                                + " spw "
                                + str(s)
                                + " poln "
                                + polstr
                                + " completely flagged in scan "
                                + str(scans[i])
                                + " due to insufficient unflagged"
                                + " data after manual flagging or abnormal negative switched power data.",
                                "WARN",
                            )

                        if calcTcals:
                            indx = np.where(data1[i, a, s, 0, :, 2] > 0)[0]
                            if len(indx) > 0:
                                funcp = np.poly1d(
                                    np.ma.polyfit(
                                        data1[i, a, s, p, indx, 0],
                                        data1[i, a, s, p, indx, 2],
                                        porder,
                                    )
                                )
                                data1[i, a, s, p, :, 3] = funcp(zmax) - funcp(zmin)
                                # from matplotlib import pyplot as plt
                                # i=0;a=20;s=19
                                # indx = np.where(data1[i,a,s,0,:,2]>0)[0]
                                # funcp = np.poly1d(np.ma.polyfit(data1[i,a,s,p,indx,0],data1[i,a,s,p,indx,2],porder))
                                # plt.plot(data1[i,a,s,p,indx,0],data1[i,a,s,p,indx,2],'b.-',data1[i,a,s,p,indx,0],funcp(data1[i,a,s,p,indx,0]),'r.-')

    tb.close()

    ## proceed with nominated solution type
    casalog.post("--> Calculating opacities and system temperature contributions.")
    if calcTcals:
        casalog.post(
            "    Zenith opacities (tau0), ant+elec contributions (Tae=Tant+Trx1+Trx2), and Tcal_new with % change from Tcal_MS are reported below."
        )
        casalog.post(
            "    Results will be highlighted if the abs(change) in Tcal_new for R or L is >= "
            + "{:.0f}".format(Tdifthresh)
            + "%.  Check Tcal solutions carefully."
        )
    else:
        casalog.post(
            "    Zenith opacities (tau0) and ant+elec contributions (Tae=Tant+Trx1+Trx2) are reported below."
        )

    dataTae = np.zeros([lenScans, lenAnt, lenSpw, 2])
    if (not calcTcals) and (tauPerAnt):
        #
        # OPTION 1: solve for opacity per scan, antenna, and spw (combined solve over both polarizations)
        #
        tb.open(caltableZ, nomodify=False)
        dataopZ = np.zeros([lenScans, lenAnt, lenSpw])
        for i in range(lenScans):
            # casalog.post('--> Processing scan '+str(scans[i])+' ('+str(i+1)+'/'+str(lenScans)+')')
            for a in range(lenAnt):
                # casalog.post('    processing antenna '+antNames[a]+' ('+str(a+1)+'/'+str(lenAnt)+')')
                for s in range(lenSpw):
                    # casalog.post('        processing spectral window '+str(s)+' ('+str(s+1)+'/'+str(lenSpw)+')')
                    # can expect that flagging will be polarization independent
                    # Tsys in data1 can be 0 (dummy value in array) or flagged (NaN)
                    # only process if valid Tsys solutions are available in, say, pol=0
                    indx = np.where(data1[i, a, s, 0, :, 2] > 0)[0]
                    if len(indx) > 0:
                        # for this scan, ant, spw: we have 2 datasets (Tsys vs ZA for 2 pols)
                        # and the equations have 3 unknowns (T0_pol1, T0_pol2, tau0)

                        Trab = get_Trab(spwCntFreq[s])
                        Tuab = get_Tuab(spwCntFreq[s])
                        # convert kinetic Twmt to noise temp in K
                        # hmm, to simplify, take mean weighted mean atmospheric temperature during scan
                        Twmtp = k2nt(np.mean(data1[i, a, s, 0, indx, 1]), spwCntFreq[s])

                        # starting estimate for unknown parameters (T0_pol1, T0_pol2, tau0)
                        se = [50.0, 50.0, 0.2]
                        fit, ier = scipy.optimize.leastsq(
                            err_multi_wrap(Trab, Tuab, Twmtp),
                            se,
                            args=(
                                data1[i, a, s, 0, indx, 0],
                                data1[i, a, s, 0, indx, 2],
                                data1[i, a, s, 1, indx, 2],
                            ),
                        )

                        # print fit
                        # x  = data1[i,a,s,0,indx,0]
                        # y1 = data1[i,a,s,0,indx,2]
                        # y2 = data1[i,a,s,1,indx,2]
                        # from matplotlib import pyplot as plt
                        # plt.plot(x,y1,'b.-',x,func(x,np.r_[fit[0],fit[-1]],Trab,Tuab,Twmtp),'r.-',
                        #         x,y2,'g.-',x,func(x,np.r_[fit[1],fit[-1]],Trab,Tuab,Twmtp),'y.-')
                        # plt.show()

                        dataopZ[i, a, s] = fit[-1]
                        dataTae[i, a, s, 0] = fit[0] - dataTcalMS[a, s, 0] / 2.0
                        dataTae[i, a, s, 1] = fit[1] - dataTcalMS[a, s, 1] / 2.0

                        casalog.post(
                            "    scan "
                            + str(scans[i])
                            + ", "
                            + antNames[a]
                            + ", spw "
                            + str(s)
                            + " - tau0: "
                            + "{:.3f}".format(fit[-1])
                            + ", Tae (K): "
                            + "{:.2f}".format(dataTae[i, a, s, 0])
                            + " (R), "
                            + "{:.2f}".format(dataTae[i, a, s, 1])
                            + " (L)"
                        )

                        k = i * lenAnt * lenSpw + a * lenSpw + s
                        tb.putcell("FPARAM", k, np.array([[fit[-1]]]))
                        tb.putcell("FLAG", k, np.array([[False]], dtype=bool))

        tb.flush()
        tb.close()
    else:
        #
        # OPTION 2: solve for opacity per scan and spw (combined solve over all antennas and polarizations)
        # OPTION 3: same as option 2, but also solve for Tcals
        #
        dataopZ = np.zeros([lenScans, lenSpw])
        if calcTcals:
            # 0=R, 1=L, 2=%difference R, 3=%difference L
            # diff = (new-old)/old*100
            dataTcal = np.zeros([lenScans, lenAnt, lenSpw, 4])

        for i in range(lenScans):
            casalog.filter("WARN")
            msmd.open(msname)
            scanspws = msmd.spwsforscan(scans[i])
            msmd.done()
            casalog.filter("INFO")
            for s in scanspws:
                # store data in prep for fitting
                dataZA = ()
                dataTsys = ()
                se = []
                AntArr = []
                getTruw = True
                if calcTcals:
                    dTsysp_median = np.nanmedian(data1[i, :, s, :, 0, 3])
                for a in range(lenAnt):
                    indx = np.where(data1[i, a, s, 0, :, 2] > 0)[0]
                    if len(indx) > 0:
                        # only need to get Trab,Tuab,Twmtp once, same for all antennas
                        # Unlikely that temperature changes much over 2 mins, so
                        # don't worry about potential flagging differences between ants
                        if getTruw:
                            Trab = get_Trab(spwCntFreq[s])
                            Tuab = get_Tuab(spwCntFreq[s])
                            Twmtp = k2nt(
                                np.mean(data1[i, a, s, 0, indx, 1]), spwCntFreq[s]
                            )
                            getTruw = False

                        if calcTcals:
                            for p in range(2):
                                C = dTsysp_median / data1[i, a, s, p, 0, 3]
                                dataTsys += (C * data1[i, a, s, p, indx, 2],)
                                dataTcal[i, a, s, p] = C * dataTcalMS[a, s, p]
                                dataTcal[i, a, s, p + 2] = (C - 1) * 100.0
                        else:
                            dataTsys += (
                                data1[i, a, s, 0, indx, 2],
                                data1[i, a, s, 1, indx, 2],
                            )

                        dataZA += (data1[i, a, s, 0, indx, 0],)
                        se += [50.0, 50.0]
                        AntArr += [a]

                se += [0.2]
                fit, ier = scipy.optimize.leastsq(
                    err_multi_wrap(Trab, Tuab, Twmtp), se, args=dataZA + dataTsys
                )

                # import matplotlib.pyplot as plt
                # a=4
                # p=0; plt.plot(dataZA[a],dataTsys[2*a+p],'b.-',dataZA[a],func(dataZA[a],[fit[2*a+p],fit[-1]],Trab,Tuab,Twmtp),'r-')
                # p=1; plt.plot(dataZA[a],dataTsys[2*a+p],'b.-',dataZA[a],func(dataZA[a],[fit[2*a+p],fit[-1]],Trab,Tuab,Twmtp),'r-')

                dataopZ[i, s] = fit[-1]
                m = 0
                for a in AntArr:
                    logpriority = "INFO"
                    if calcTcals:
                        dataTae[i, a, s, 0] = fit[2 * m] - dataTcal[i, a, s, 0] / 2.0
                        dataTae[i, a, s, 1] = (
                            fit[2 * m + 1] - dataTcal[i, a, s, 1] / 2.0
                        )
                        extrastr = (
                            ", Tcal_new (K): "
                            + "{:.2f}".format(dataTcal[i, a, s, 0])
                            + " (R), "
                            + "{:.2f}".format(dataTcal[i, a, s, 1])
                            + " (L), "
                            + "% change from Tcal_MS: "
                            + "{:.1f}".format(dataTcal[i, a, s, 2])
                            + " (R), "
                            + "{:.1f}".format(dataTcal[i, a, s, 3])
                            + " (L)"
                        )
                        if (np.abs(dataTcal[i, a, s, 2]) >= Tdifthresh) or (
                            np.abs(dataTcal[i, a, s, 3]) >= Tdifthresh
                        ):
                            logpriority = "WARN"
                    else:
                        dataTae[i, a, s, 0] = fit[2 * m] - dataTcalMS[a, s, 0] / 2.0
                        dataTae[i, a, s, 1] = fit[2 * m + 1] - dataTcalMS[a, s, 1] / 2.0
                        extrastr = ""

                    casalog.post(
                        "    scan "
                        + str(scans[i])
                        + ", spw "
                        + str(s)
                        + ", "
                        + antNames[a]
                        + " - tau0: "
                        + "{:.3f}".format(fit[-1])
                        + ", Tae (K): "
                        + "{:.2f}".format(dataTae[i, a, s, 0])
                        + " (R), "
                        + "{:.2f}".format(dataTae[i, a, s, 1])
                        + " (L)"
                        + extrastr,
                        logpriority,
                    )
                    m += 1
                    k = i * lenAnt * lenSpw + a * lenSpw + s
                    tb.open(caltableZ, nomodify=False)
                    tb.putcell("FPARAM", k, np.array([[fit[-1]]]))
                    tb.putcell("FLAG", k, np.array([[False]], dtype=bool))
                    tb.flush()
                    tb.close()
                    if calcTcals:
                        tb.open(caltableT, nomodify=False)
                        tb.putcell(
                            "NOISE_CAL",
                            k,
                            np.array(
                                [
                                    [dataTcal[i, a, s, 0], dataTcal[i, a, s, 1]],
                                    [0.0, 0.0],
                                ]
                            ),
                        )
                        tb.flush()
                        tb.close()

    # print out summary statistics
    casalog.post(
        "--> Print summary statistics for zenith opacity (over antenna) in nepers: "
    )
    if not calcTcals:
        if tauPerAnt:
            #
            # OPTION 1: opacity was solved per scan, antenna, and spw
            #
            casalog.post(
                "    median, median absolute deviation, min outlier, max outlier."
            )
            for i in range(lenScans):
                casalog.post("    scan " + str(scans[i]) + ":")
                casalog.filter("WARN")
                msmd.open(msname)
                scanspws = msmd.spwsforscan(scans[i])
                msmd.done()
                casalog.filter("INFO")
                for s in scanspws:
                    # value of zero could be present if all data was flagged
                    # don't let this contribute to statistics
                    sA = dataopZ[i, :, s]
                    sB = sA[np.abs(dataopZ[i, :, s]) > 0]
                    s1 = np.median(sB)
                    s2 = np.median(np.abs(sB - s1))
                    s3 = np.min(sB)
                    s4 = np.max(sB)
                    casalog.post(
                        "      spw "
                        + str(s)
                        + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                        + "{:6.3f}".format(s1)
                        + ", "
                        + "{:6.3f}".format(s2)
                        + ", "
                        + "{:6.3f}".format(s3)
                        + ", "
                        + "{:6.3f}".format(s4)
                    )
        else:
            #
            # OPTION 2: opacity was solved per scan and spw
            #
            for i in range(lenScans):
                casalog.post("    scan " + str(scans[i]) + ":")
                casalog.filter("WARN")
                msmd.open(msname)
                scanspws = msmd.spwsforscan(scans[i])
                msmd.done()
                casalog.filter("INFO")
                for s in scanspws:
                    casalog.post(
                        "      spw "
                        + str(s)
                        + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                        + "{:6.3f}".format(dataopZ[i, s])
                    )
    else:
        #
        # OPTION 3: same as option 2
        #
        for i in range(lenScans):
            casalog.post("    scan " + str(scans[i]) + ":")
            casalog.filter("WARN")
            msmd.open(msname)
            scanspws = msmd.spwsforscan(scans[i])
            msmd.done()
            casalog.filter("INFO")
            for s in scanspws:
                casalog.post(
                    "      spw "
                    + str(s)
                    + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                    + "{:6.3f}".format(dataopZ[i, s])
                )

    casalog.post(
        "--> Print summary statistics for Tae (over antenna and polarization) in K: "
    )
    casalog.post("    median, median absolute deviation, min outlier, max outlier")
    for i in range(lenScans):
        casalog.post("    scan " + str(scans[i]) + ":")
        casalog.filter("WARN")
        msmd.open(msname)
        scanspws = msmd.spwsforscan(scans[i])
        msmd.done()
        casalog.filter("INFO")
        for s in scanspws:
            # value of zero could be present if all data was flagged
            # don't let this contribute to statistics
            sA = dataTae[i, :, s, :]
            sB = sA[np.abs(dataTae[i, :, s, :]) > 0]
            s1 = np.median(sB)
            s2 = np.median(np.abs(sB - s1))
            s3 = np.min(sB)
            s4 = np.max(sB)
            casalog.post(
                "      spw "
                + str(s)
                + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                + "{:8.3f}".format(s1)
                + ", "
                + "{:8.3f}".format(s2)
                + ", "
                + "{:8.3f}".format(s3)
                + ", "
                + "{:8.3f}".format(s4)
            )

    if calcTcals:
        casalog.post(
            "--> Print summary statistics for new Tcal solutions (over antenna and polarization) in K: "
        )
        casalog.post("    median, median absolute deviation, min outlier, max outlier")
        casalog.post("    scan " + str(scans[i]) + ":")
        casalog.filter("WARN")
        msmd.open(msname)
        scanspws = msmd.spwsforscan(scans[i])
        msmd.done()
        casalog.filter("INFO")
        for s in scanspws:
            # value of zero could be present if all data was flagged
            # don't let this contribute to statistics
            sA = dataTcal[i, :, s, 0:2]
            sB = sA[np.abs(dataTcal[i, :, s, 0:2]) > 0]
            s1 = np.median(sB)
            s2 = np.median(np.abs(sB - s1))
            s3 = np.min(sB)
            s4 = np.max(sB)
            casalog.post(
                "      spw "
                + str(s)
                + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                + "{:8.3f}".format(s1)
                + ", "
                + "{:8.3f}".format(s2)
                + ", "
                + "{:8.3f}".format(s3)
                + ", "
                + "{:8.3f}".format(s4)
            )

        casalog.post(
            "--> Print summary statistics for dTcal(%) = (Tcal_new-Tcal_ref)/Tcal_ref*100 (over antenna and polarization): "
        )
        casalog.post("    median, median absolute deviation, min outlier, max outlier")
        casalog.post("    scan " + str(scans[i]) + ":")
        casalog.filter("WARN")
        msmd.open(msname)
        scanspws = msmd.spwsforscan(scans[i])
        msmd.done()
        casalog.filter("INFO")
        for s in scanspws:
            # value of zero could be present if all data was flagged
            # don't let this contribute to statistics
            sA = dataTcal[i, :, s, 2:4]
            sB = sA[np.abs(dataTcal[i, :, s, 2:4]) > 0]
            s1 = np.median(sB)
            s2 = np.median(np.abs(sB - s1))
            s3 = np.min(sB)
            s4 = np.max(sB)
            casalog.post(
                "      spw "
                + str(s)
                + " ({:.4f} GHz): ".format(spwCntFreq[s] / 1e9)
                + "{:8.3f}".format(s1)
                + ", "
                + "{:8.3f}".format(s2)
                + ", "
                + "{:8.3f}".format(s3)
                + ", "
                + "{:8.3f}".format(s4)
            )
