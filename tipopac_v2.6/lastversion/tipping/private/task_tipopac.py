from __future__ import absolute_import
# get is_CASA6 and is_python3
from casatasks.private.casa_transition import is_CASA6

if is_CASA6:
    from casatasks import casalog
    import casatools
else:
    import casac
    from taskinit import *

#sys.path.append('/lustre/aoc/sciops/pbeaklin/allin/short/TestTask/Temp/analysis_scripts') 
#import analysisUtils as au


import numpy as np
import math
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import linalg
import matplotlib.pyplot as plt
from warnings import simplefilter
import os




# tipopac is released under a BSD 3-Clause License
# See LICENSE for details

# HISTORY:
#   1.0  22Oct2019  Initial version.
#



def tipopac(msname,caltableZ,tauPerAnt,calcTcals,caltableT,cmdFlag,usrFlag,flagFile,caltable,doPlot,doModel):
    simplefilter(action='ignore',category=FutureWarning)
    simplefilter(action='ignore',category=RuntimeWarning)

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
    
    casalog.origin('tipopac')
    casalog.post('--> tipopac version 2.5')

    # JVLA tips runs between approx 55-23 deg elevation (35-67 deg zenith angle, respectively)
    porder = 3    # order of polynomial fit
    zmin   = 40.  # zenith angle min, degrees
    zmax   = 62.  # zenith angle max, degrees
    
    # minimum number of (assuming 1 sec) integration times to get a good solution
    # pick ~40 sec somewhat randomly.  A normal tipping scan should take ~90 sec.
    minTipInts = 3
    
    # for calcTcals=True, set threshold warning level for % diff with TcalMS
    Tdifthresh = 30
    #Pedro added next if
    if caltableZ=='': caltableZ = 'defaultZ'
    if caltable and caltableZ == 'defaultZ':
                casalog.post("*** WARNING: caltable = True, but no name to opacity caltable. Set as defaultZ.cal.","WARN")   
    if caltableT=='':    
        if calcTcals:
                casalog.post("*** WARNING: caltable = True, but no name to Tcal caltable. Set as defaultT.cal","WARN")
                caltableT='defaultT.cal'
    else:
        if calcTcals==False and caltableT!='defaultT':
                casalog.post("*** WARNING: caltableT=False. File "+caltableT+" will not be saved.","WARN")
    
    if calcTcals & tauPerAnt:
        casalog.post("*** WARNING: Setting tauPerAnt=False because calcTcals=True.","WARN")
        tauPerAnt = False
    #if caltable and tauPerAnt: 
    #            casalog.post("*** WARNING: caltable = True, getting tau per antenna not allowed. Set tauPerAnt = False","WARN")
    #            tauPerAnt = False
    if not caltable and caltableZ != 'defaultZ': 
                casalog.post("INFO: caltable = False, opacity calibration table will not be saved.","INFO")
    if not caltable and caltableZ != 'defaultT': 
                casalog.post("INFO: caltable = False, Tcal calibration table will not be saved.","INFO") 
    if doPlot:
        if os.path.exists(msname+'.tipping.plots/'):
                 casalog.post("INFO: Plots will be saved at "+msname+".tipping.plots","INFO")
        else:
                 os.system('mkdir '+msname+'.tipping.plots')
                 casalog.post("INFO: Plots will be saved at "+msname+".tipping.plots","INFO")
    
    # avoid using global cb tool, which can cause issues with table cache
    def gencaltableZ(msname,caltableZ):
        if is_CASA6:
           mycb = casatools.calibrater()
        else:
           mycb = casac.calibrater()
        mycb.open(msname,False,False,False)
        mycb.createcaltable(caltableZ,'Real','TOpac',True)
        mycb.close()
    
    
    # kinetic to noise temp in K
    def k2nt(T,nu_Hz):
        h = 6.6261e-34
        k = 1.3806e-23
        return T * (h*nu_Hz/(k*T)/(np.exp(h*nu_Hz/(k*T))-1.))
    

    
    # Tsys vs z tipping curve function
    def func(z,params,Twmtp):
        # z in deg, T's all in noise K
        # T0 = Tant + Trx1 + Trx2 + Tcal/2 ~= constant
        T0,tau0 = params
        Tsys = T0+Twmtp*(1-np.exp(-tau0/np.cos(np.deg2rad(z))))
        return Tsys
    
    # option 1: calcTcals=False and tauPerAnt=True
    #           3 unknown parameters: T0_pol0, T0_pol1, tau0
    # option 2: calcTcals=False and tauPerAnt=False
    #           2*nant+1 unknown parameters:
    #             T0_a0_p0, T0_a0_p1, ..., T0_aN_p0, T0_aN_p1, tau0
    # option 3: same as 2, but Tsys values will be adjusted beforehand
    # these can share the same wrapper (op1: N=1, op2/3: N=nant)

    def err_multi_wrap(Twmtp):
        def err_multi(p,*argv):  
            N      = int(len(argv)/3)
            z      = argv[:N]
            Tsys   = argv[N:]
            params = p[0],p[-1]
            errArr = Tsys[0]-func(z[0],params,Twmtp)
            for k in range(1,2*N):
                params = p[k],p[-1]
                errArr = np.concatenate([errArr,Tsys[k]-func(z[int(k/2)],params,Twmtp)])
            return errArr
        return err_multi

    def err_multi_wrapTcal(Twmtp):
    	    	def err_multiTcal(p,*argv):  
    	        	N      = int(len(argv)/3)
    	        	z      = argv[:N]
    	        	Tsys   = argv[N:]
    	        	params = p[0],p[-1]
    	        	errArr = Tsys[0]-((p[0]+Twmtp*(1-np.exp(-p[-1]/np.cos(np.deg2rad(z[0])))))/p[1])
    	        	inde = 1
    	        	for k in range(2,4*N,2):
    	        	    	errArr = np.concatenate([errArr,Tsys[inde]-((p[k]+Twmtp*(1-np.exp(-p[-1]/np.cos(np.deg2rad(z[int(inde/2)])))))/p[k+1])])
    	        	    	inde += 1
    	        	return errArr
    	    	return err_multiTcal



    def fitting_Tcal(arg,seT,bound1,bound2,bound3,bound4,bound5,bound6,Twmtp):
        def compute_rms(model,argv,Twmtp):
                        N      = int(len(argv)/3)
                        z      = argv[:N]
                        Tsys   = argv[N:]
                        errArr = Tsys[0]-((model[0]+Twmtp*(1-np.exp(-model[-1]/np.cos(np.deg2rad(z[0])))))/model[1])
                        inde = 1
                        for k in range(2,4*N,2):
                                errArr = np.concatenate([errArr,Tsys[inde]-((model[k]+Twmtp*(1-np.exp(-model[-1]/np.cos(np.deg2rad(z[int(inde/2)])))))/model[k+1])])
                                inde += 1
                        errArr = np.array(errArr)
                        return (np.sum(errArr**2))/(float(len(errArr))-float(4*N+1))
        Nok = True
        try:
                fitall = scipy.optimize.least_squares(err_multi_wrapTcal(Twmtp),seT,args=arg,bounds=(bound1,bound2))
                Upper = 0.98*np.array(bound2)-np.array(fitall.x)
                Lower = 1.02*np.array(bound1)-np.array(fitall.x)
                Nupper = np.sum(np.array(Upper) < 0)
                Nlower = np.sum(np.array(Lower) > 0)
                VersionFit = 1
                #print(VersionFit,fitall[-1])
                if (Lower[-1] > 0) or (Upper[-1] < 0) or (Nupper+Nlower > 8): 
                                fitall = scipy.optimize.least_squares(err_multi_wrapTcal(Twmtp),seT,args=arg,bounds=(bound3,bound4),max_nfev=1000)
                                Upper = 0.98*np.array(bound3)-fitall.x
                                Lower = 1.02*np.array(bound4)-fitall.x
                                Nupper = np.sum(np.array(Upper) < 0)
                                Nlower = np.sum(np.array(Lower) > 0)
                                #print(VersionFit,fitall[-1])
                                if (Lower[-1] > 0) or (Upper[-1] < 0) or (Nupper+Nlower > 10): 
                                                fitall = scipy.optimize.least_squares(err_multi_wrapTcal(Twmtp),seT,args=arg,bounds=(bound5,bound6),max_nfev=1000)
                                                Upper = 0.98*np.array(bound5)-fitall.x
                                                Lower = 1.02*np.array(bound6)-fitall.x
                                                VersionFit = 3
                                                #print(VersionFit,fitall[-1])
				
        except:
                Nok = False
                VersionFit = 0
                Upper = np.zeros(len(bound1))-1
                Lower = np.zeros(len(bound1))+1             

        if Nok:
                FIT = np.array(fitall.x)
                jac = np.array(fitall.jac)
                RMS = compute_rms(FIT,arg,Twmtp)
                try:
                        COV = linalg.inv(jac.T.dot(jac))
                        Err = np.sqrt(np.diag(RMS * RMS * COV))
                except:
                        Err= 0.8*FIT
        else:
                FIT = np.zeros(len(seT)+1)-1
                RMS = -1
                Err = FIT

        indexBad = (np.array(Upper) < 0) + (np.array(Lower) > 0)

		
        Nant = int((len(FIT)-1)/2)
        fit = np.zeros(Nant+1)
        Tcal = np.zeros(Nant)
        efit = np.zeros(Nant+1)
        eTcal = np.zeros(Nant)
        fit[-1] = FIT[-1]
        efit[-1] = Err[-1]
        ind =0
        for i in range(Nant):
                fit[i] = FIT[ind]
                Tcal[i] = FIT[ind+1]
                efit[i] = Err[ind]
                eTcal[i] = Err[ind+1]
                ind += 2



        return fit, Tcal, efit, eTcal, indexBad, VersionFit

    def makeplot(antennaName,spw,ZA,tsysRight,tsysLeft,trRight,trLeft,tcalRight,tcalLeft,tau,etau,eTR,eTL,eTcalR,eTcalL,Twmtp,scanstring):
                if np.min(tsysRight) < np.min(tsysLeft):
                	limY1 = 0.9*np.min(tsysRight)
                else:
                	limY1 = 0.9*np.min(tsysLeft)
                if np.max(tsysRight) > np.max(tsysLeft):
                	limY2 = 1.1*np.max(tsysRight)
                else:
                	limY2 = 1.1*np.max(tsysLeft)
                plt.ioff()
                fig = plt.figure()
                if caltable:
                    plt.title(antennaName+' at spw '+str(spw)+': Tau = '+'{:.3f}'.format(tau)+' pm '+'{:.3f}'.format(etau)+'\n'+'TR= '+'{:.1f}'.format(trRight)+' pm '+'{:.1f}'.format(eTR)+' TL= '+'{:.1f}'.format(trLeft)+' pm '+'{:.1f}'.format(eTL)+' TcalR= '+'{:.2f}'.format(tcalRight)+' pm '+'{:.2f}'.format(eTcalR)+' TcalL= '+'{:.2f}'.format(tcalLeft)+' pm '+'{:.2f}'.format(eTcalL))
                else:
                    plt.title(antennaName+' at spw '+str(spw)+': Tau = '+'{:.3f}'.format(tau)+' pm '+'{:.3f}'.format(etau))
                plt.xlim(30,90)
                try:
                     plt.ylim(limY1,limY2)
                     plt.ylabel('Tsys(K)', fontsize=16)
                     plt.xlabel('Zenith Angle', fontsize=16)
                     plt.scatter(ZA, tsysRight,color='blue',label='R Pol')
                     plt.scatter(ZA, tsysLeft,color='green',label='L Pol')
                     modelR = (trRight + Twmtp*(1-np.exp(-tau/np.cos(np.deg2rad(ZA)))))/tcalRight
                     modelL = (trLeft +Twmtp*(1-np.exp(-tau/np.cos(np.deg2rad(ZA)))))/tcalLeft
                     plt.plot(ZA,modelR,color='red',label='fit')
                     plt.plot(ZA,modelL,color='red')
                     plt.legend()
                     if not tauPerAnt:
                          plt.savefig(msname+'.tipping.plots/tippingcurve_spw_'+str(spw)+'_'+antennaName+'_scan_'+scanstring+'.png')
                     else:
                          plt.savefig(msname+'.tipping.plots/tippingcurve_spw_'+str(spw)+'_'+antennaName+'_scan_'+scanstring+'.perantenna.png')
                     plt.close(fig)
                except:
                     plt.close(fig)
    #make plot per antenna		
    def water_saturation_pressure (temperature, pressure):
        """calculate saturation vapor pressure over water or ice, in mbar,
           given temperature in degrees C, and pressure in mbar.  from:
    
           Buck, A., New Equations for Computing Vapor Pressure and
              Enhancement Factor, J. Appl. Met., v.20, pp. 1527-1532, 1981
    
           who references:
    
           Wexler, A., Vapor Pressure Formulation for Water in the Range
              0C to 100C - A Revision, J. Res. Natl. Bur. Stand., v.80A,
              pp. 775 ff, 1976
    
           and
    
           Wexler, A., Vapor Pressure Formulation for Ice, J. Res. Natl.
              Bur. Stand., v.81A, pp. 5-20, 1977
    
           for the "exact" formulations - which are reworks of the
           Goff-Gratch formulation:
    
           Goff, J.A., and S. Gratch, Low-pressure Properties of Water from
              -160F to 212F. Trans. Am. Soc. Heat. Vent. Eng., v. 52, 95-121,
              1946
    
           Use the fw5 and fi5 coefficients for the "enhancement
           factor" from Table 3 of Buck."""
    
        A_w = 4.1e-4
        B_w = 3.48e-6
        C_w = 7.4e-10
        D_w = 30.6e0
        E_w = -3.8e-2
        A_i = 4.8e-4
        B_i = 3.47e-6
        C_i = 5.9e-10
        D_i = 23.8e0
        E_i = -3.1e-2
    
        theta = temperature + 273.15e0
    
        if temperature > 0.01e0:
    #
    # water
    #
            ew = (-2991.2729e0 / theta**2.0e0) + (-6017.0128 / theta) + \
                 (18.87643854e0) + (-0.028354721e0 * theta) + \
                 (0.17838301e-4 * theta**2.0e0) + \
                 (-0.84150417e-9 * theta**3.0e0) + \
                 (0.44412543e-12 * theta**4.0e0) + \
                 (2.858487 * np.log(theta))
            ew = 0.01e0 * np.exp (ew)
            fw = 1.0e0 + A_w + \
                 pressure * (B_w + C_w * (temperature + D_w + E_w * pressure)**2.0e0)
            saturation_pressure = ew * fw
        else:
    #
    # ice
    #
            ei = (-5865.3696e0 / theta) + \
                 (22.241033e0) + (0.013749042e0 * theta) + \
                 (-0.34031775e-4 * theta**2.0e0) + \
                 (0.26967687e-7 * theta**3.0e0) + \
                 (0.6918651 * np.log(theta))
            ei = 0.01e0 * np.exp (ei)
            fi = 1.0e0 + A_i + \
                 pressure * (B_i + C_i * (temperature + D_i + E_i * pressure)**2.0e0)
            saturation_pressure = ei * fi
        return saturation_pressure

    def createCasaTool(mytool):
        """
        A wrapper to handle the changing ways in which casa tools are invoked.
        For CASA < 6, it relies on "from taskinit import *" in the preamble above.
        mytool: a tool name, like tbtool
        Todd Hunter
        """
        if 'casac' in locals():
        	if (type(casac.Quantity) != type):  # casa 4.x and 5.x
        		myt = mytool()
        	else:  # casa 3.x
        		myt = mytool.create()
        else:
        	# this is CASA 6
        	myt = mytool()
        return(myt)

    def create_casa_quantity(myqatool,value,unit):
        """
        A wrapper to handle the changing ways in which casa quantities are invoked.
        myqatool: an existing instance of the qa tool
        value: value to set
        unit: unit to set
        Todd Hunter
        """
        if 'casac' in locals():
        	if (type(casac.Quantity) != type):  # casa 4.x and 5.x
        		myqa = myqatool.quantity(value, unit)
        	else:  # casa 3.x
        		myqa = casac.Quantity(value, unit)
        else:
        	# This is CASA 6 (same as 4.x and 5.x)
        	myqa = myqatool.quantity(value,unit)

        return(myqa)

    def getAtmDetails(myat):
        """
        A wrapper to handle the changing ways in which the at tool is accessed.
        Todd Hunter
        """
        if 'casac' in locals():
        	if (type(casac.Quantity) == type):  # casa 3.x
        		dry = np.array(myat.getDryOpacitySpec(0)['dryOpacity'])
        		wet = np.array(myat.getWetOpacitySpec(0)['wetOpacity'].value)
        		TebbSky = []
        		n = myat.getNumChan()
        		for chan in range(n):  # do NOT use numchan here, use n
        			TebbSky.append(myat.getTebbSky(nc=chan, spwid=0).value)
        		TebbSky = np.array(TebbSky)
        		# readback the values to be sure they got set
        		rf = myat.getRefFreq().value
        		cs = myat.getChanSep().value
        	else:  # casa 4.x and 5.x
	        	dry = np.array(myat.getDryOpacitySpec(0)[1])
	        	wet = np.array(myat.getWetOpacitySpec(0)[1]['value'])
	        	TebbSky = myat.getTebbSkySpec(spwid=0)[1]['value']
	        	# readback the values to be sure they got set
        		rf = myat.getRefFreq()['value']
        		cs = myat.getChanSep()['value']
        else:
        	# this is CASA 6 (same as 4.x and 5.x)
        	dry = np.array(myat.getDryOpacitySpec(0)[1])
        	wet = np.array(myat.getWetOpacitySpec(0)[1]['value'])
        	TebbSky = myat.getTebbSkySpec(spwid=0)[1]['value']
        	# readback the values to be sure they got set
        	rf = myat.getRefFreq()['value']
        	cs = myat.getChanSep()['value']
        return(dry,wet,TebbSky,rf,cs)


    def estimateOpacity(pwvmean=1.0,reffreq=230,conditions=None,verbose=True,
                    elevation=90, altitude=5059, P=563, H=20, T=273, Trx=0,
                    etaTelescope=0.75, telescope=None, airmass=0, dP=5.0, h0=1.0,
                    maxAltitude=48.0, dPm=1.1):
        """
        Estimate the opacity at a specified frequency and weather condition at an
        observatory using J. Pardo's ATM in casa.
        Return values:
        If Trx is not specified, then it returns:  [tauZenith, tau]
        If Trx is specified, then it returns:
        [[tauZenith, tau], [transZenith, trans], [TskyZenith, Tsky], [TsysZenith, Tsys]]
        Equation for Tsys:
        (Trx + etaTelescope*T*(1-etaSky) + T*(1-etaTelescope))/(etaTelescope*etaSky)
        where etaSky = atmospheric transmission (as a fraction)
        and etaTelescope = telescope efficiency (as a fraction)
        units: pwv(mm), reffreq(GHz), elevation(deg), altitude(m), P=pressure(mb), T=temp(K),
        H=relativeHumidity(percent)
        h0: scale height (in km)
        maxAltitude: of atmosphere (in km)
        dP: pressure step in the model, has units of pressure (mb)
        dPm: pressure step factor in the model (unitless) called PstepFact in TelCal
        The default values are nominal conditions at ALMA.  To change them, you may
        either specify the weather and location variables, or use the shortcuts:
         telescope: 'ALMA', 'SMA', 'EVLA'
        conditions: a dictionary containing keys and values for (at least):
        'pressure','temperature','humidity','elevation'
        For further help, see
        https://safe.nrao.edu/wiki/bin/view/ALMA/EstimateOpacity
        -- Todd Hunter
        """

        if (airmass >= 1.0):
        	elevation = math.asin(1./airmass)*180/math.pi
        if (conditions is not None):
        	if (conditions['pressure'] > 1e-10):
        		#       angle = conditions['solarangle'] # unused
        		P = conditions['pressure']
        		T = celsiusToKelvin(conditions['temperature'])
        		H = conditions['humidity']
        		elevation = conditions['elevation']
        elif (telescope is not None):
        	# default P,H,T are set to ALMA typical, but should get this from observatory
        	if (telescope == 'SMA'):
        		P = 629.5
        		altitude = 4072
        		print("Using pressure=%.1fmb, temperature=%.1fK and humidity=%.0f%% at SMA." % (P,T,H))
        elif (telescope == 'VLA'):
        	P = 785
        	altitude = 2124
        	print("Using pressure=%.1fmb, temperature=%.1fK and humidity=%.0f%% at %s." % (P,T,H,telescope))
        elif (telescope == 'ALMA'):
        	print("Unrecognized telescope.  Available choices: SMA, ALMA, (E)VLA")
        	return
        else:
        	if (verbose):
        		print("Using pressure=%.1fmb, temperature=%.1fK, humidity=%.0f%%, etaTelescope=%.2f at ALMA." % (P,T,H,etaTelescope))
        chansep=1
        numchan=1
        nb = 1
        reffreq = np.double(reffreq)
        chansep = np.double(chansep)
        numchan = int(numchan)
        chansep = np.double(chansep)
        spwid = 0
        if not is_CASA6:
        	myqa = createCasaTool(qatool)
        else:
        	myqa = casatools.quanta()
        try:
        	if not is_CASA6:
        		myat = createCasaTool(attool)
        	else:
        		myat = casatools.atmosphere()
        	needToCloseAT = True
        except:  # CASA < 5.0.0
        	needToCloseAT = False
        	myat = at
        myat.initAtmProfile(humidity=H,
                     temperature=create_casa_quantity(myqa, T,"K"),
                     altitude=create_casa_quantity(myqa, altitude,"m"),
                     pressure=create_casa_quantity(myqa, P,'mbar'),
                     atmType = 1,
                     h0 = create_casa_quantity(myqa, h0,"km"),
                     maxAltitude = create_casa_quantity(myqa, maxAltitude,"km"),
                     dP = create_casa_quantity(myqa, dP,"mbar"),
                     dPm=dPm)
        fC = create_casa_quantity(myqa, reffreq,'GHz')
        fR = create_casa_quantity(myqa, chansep,'GHz')
        fW = create_casa_quantity(myqa, numchan*chansep,'GHz')
        myat.initSpectralWindow(nb,fC,fW,fR)
        myat.setUserWH2O(create_casa_quantity(myqa, pwvmean,'mm'))
        dry, wet, TebbSkyZenith, rf, cs = getAtmDetails(myat)
        if needToCloseAT:
        	myat.close()
        transZenith = math.exp(-dry-wet)*100
        if (verbose):
        	print("%5.1f GHz tau at zenith= %.3f (dry=%.3f,wet=%.3f), trans=%.1f%%, Tsky=%.2f" % (reffreq,dry+wet, dry, wet,math.exp(-dry-wet)*100, TebbSkyZenith))
        airmass = 1/math.sin(elevation*math.pi/180.)
        TebbSky = TebbSkyZenith * (1-np.exp(-airmass*(wet+dry)))/(1-np.exp(-wet-dry))
        trans = math.exp((-dry-wet)*airmass)*100
        if (verbose and elevation<90):
        	print("%5.1f GHz tau toward source (elev=%.1f,airm=%.2f)=%.3f, trans=%.1f%%, Tsky=%.2f" % (reffreq,elevation,airmass,(dry+wet)*airmass,math.exp((-dry-wet)*airmass)*100, TebbSky))
        if (Trx > 0):
        	# compute expected Tsys
        	etaSkyZenith = math.exp(-(dry+wet))
        	TsysZenith = (T*(1/etaSkyZenith-1) + Trx/etaSkyZenith) / etaTelescope
        	etaSky = math.exp(-airmass*(dry+wet))
        	Tsys = (T*(1/etaSky-1) + Trx/etaSky) / etaTelescope
        	if (verbose):
        		print("Expected Tsys at zenith = %.1fK,   toward source = %.1fK" % (TsysZenith,Tsys))
	        return([[dry+wet, (dry+wet)*airmass], [transZenith,trans], [TebbSkyZenith,TebbSky],[TsysZenith,Tsys]])
        else:
        	return([dry+wet, (dry+wet)*airmass])





    def model(pwv,HH,TT,PP,hs):
        F = np.zeros(501)
        tau = np.zeros(501)
        for i in range(501):
        	F[i] = 0.+0.1*i
        	vla_opacity = estimateOpacity(pwvmean=pwv,reffreq=F[i],altitude=2124,P=PP,H=HH,T=TT,h0=hs,maxAltitude=20.0,verbose=False)
        	tau[i] = vla_opacity[0]
        return F, tau


    def fitATM(frequency,opacity,eopacity,temperature,pressure,humidity,dew):
        #bestcurve = False # is that really needed? I think should be False.
        def residuals(XX,YY,FF,TT,EE):
        	residuals = np.zeros(len(FF))
        	resiW = np.zeros(len(FF))
        	NE = 1/EE
        	for j in range(len(FF)):
        		Diff = abs(XX-FF[j])
        		mindiff = np.min(Diff)
        		YYfit = YY[Diff==mindiff]
        		residuals[j] = (YYfit-TT[j])
        		resiW[j] = ((YYfit-TT[j]) * (YYfit-TT[j]))/EE[j]
        	resi = np.sum(resiW)/np.sum(NE)
        	if np.isnan(resi): resi = 1000
        	return np.sqrt(resi), residuals
        def meanW(XX,EE):
        	XX = np.array(XX)
        	EE = np.array(EE)
        	Sum = 0.
        	NN = np.sum(EE)
        	for i in range(len(XX)):
        		Sum += EE[i]*XX[i]
        	answer = Sum/NN
        	upperLimit = np.max(XX)-answer
        	lowerLimit = answer-np.min(XX)
        	if upperLimit > lowerLimit:
        		errorAnswer = upperLimit
        	else:
        		errorAnswer = lowerLimit
        	return answer, errorAnswer
        frequency = np.array(frequency)
        opacity = np.array(opacity)
        error = np.array(eopacity)
        dew = np.array(dew)
        temperature = np.array(temperature)
        pressure = np.array(pressure)
        humidity = np.array(humidity)
        P = np.mean(pressure)
        T = np.mean(temperature)
        H = np.mean(humidity)
        D = np.mean(dew)
        m_w = 18 * 1.6749286e-27
        rho_l = 1000.0
        k_0 = 1.380658e-23
        saturation_air = water_saturation_pressure(D-273.15,P)
        P_0 = 100.*saturation_air
        ListF = np.unique(frequency[frequency>0])
        Nspw = len(ListF)
        spwF0 = np.zeros(Nspw)
        spwT0 = np.zeros(Nspw)
        spwE0 = np.zeros(Nspw)
        err_frac2 = np.zeros(Nspw)
        Nk = 0
        Nk_used = 0
        for f in range(Nspw):
        	spwF0[f] = (np.mean(frequency[frequency==ListF[f]]))/1e9
        	spwT0[f] = np.mean(opacity[frequency==ListF[f]])
        	spwE0[f] = np.mean(error[frequency==ListF[f]])
        	if spwF0[f] > 45:
        	     err_frac2[f] = ((spwT0[f]*0.025)/0.1)/spwE0[f]
        	elif spwF0[f] > 18 and spwF0[f] < 26.5:
        	     err_frac2[f] = ((spwT0[f]*0.025)/0.1)/spwE0[f]
        	     Nk += 1
        	     if err_frac2[f]> 1: Nk_used += 1
        	else:
        	     err_frac2[f] = ((spwT0[f]*0.015)/0.1)/spwE0[f]
        if Nk > 0 and Nk_used < 3:
            Nk_used2 = 0
            for f in range(Nspw):
                 if spwF0[f] > 18 and spwF0[f] < 26.5: 
                           err_frac2[f] = ((spwT0[f]*0.04)/0.1)/spwE0[f]
                           Nk_used2 += 1
            if Nk_used2 < 4:
                 for f in range(Nspw):
                    if spwF0[f] > 18 and spwF0[f] < 26.5: err_frac2[f] = ((spwT0[f]*0.08)/0.1)/spwE0[f]

        spwF = spwF0[err_frac2>1]
        spwT = spwT0[err_frac2>1]
        spwE = spwE0[err_frac2>1]
        spwFFlag = spwF0[err_frac2<1]
        spwTFlag = spwT0[err_frac2<1]
        spwEFlag = spwE0[err_frac2<1]
        ### IF ALL DATA ARE FLAGGED, DO NOT FLAG AND SHOW ALL DATA
        if len(spwE) == 0:
           print(' Due to the large noise in the data, we do not recommend modeling.')
           spwF = spwF0
           spwT = spwT0
           spwE = spwE0
           spwFFlag = spwF0
           spwTFlag = spwT0
           spwEFlag = spwE0
        spwFK = spwF[(spwF > 18) * (spwF < 26.5)]
        spwTK = spwT[(spwF > 18) * (spwF < 26.5)]
        spwEK = spwE[(spwF > 18) * (spwF < 26.5)]
        #listofrms = []
        HH = 1000*2.0
        pwvpred = 1000.0 * m_w * P_0 * HH / (rho_l * k_0 * T)
        maxpwv = 3*pwvpred
        Npwv = int(3*maxpwv)
        hspwv = [1.0,1.5,2.0,2.5,3.0]
        hsrange = np.random.uniform(1,6,100)
        pwvl = np.random.uniform(0,maxpwv,Npwv)
        rmspwv = np.zeros([5,Npwv])
        indh = -1
        meanerr = (np.median(spwEK))/2.0
        casalog.post("INFO: Finding the best scale height. It may take a few minutes.","INFO")
        if (len(spwFK) > 2):
        	rmsf = 10.
        	for hs in hspwv:
        		indh += 1
        		indp = -1
        		#listofrms = []
        		for pwvi in pwvl:
        			indp += 1
        			HH = hs*1000
        			pwv0 = pwvi #1000.0 * m_w * P_0 * HH / (rho_l * k_0 * T)
        			xx,yy = model(pwv0,H,T,P,hs)
        			rms, resi = residuals(xx,yy,spwFK,spwTK,spwEK)
        			rmspwv[indh,indp] = rms
        			#listofrms.append(rms)
        			#minresi = np.min(listofrms)
        			if rms < rmsf: 
        					pwvf = pwvi
        					hf = hs
        					rmsf = rms
        					indhf = indh
        			#print(indh,indp,hs,pwvi,rms,rmsf,hf,pwvf)
        	listrmspwv = rmspwv[indhf]
        	#listofrms = np.array(listofrms)
        	#listrmsspwv2 = listrmsspwv[listrmsspwv<1.25*rmsf]
        	possiblepwv = pwvl[listrmspwv<1.25*rmsf]
        	#print(len(possiblepwv))
        	#print(possiblepwv)
        	if len(possiblepwv) == 1:
        		epwv = 0.25*pwvf
        	else:
        		epwv = np.std(possiblepwv)
        	#minresi = np.min(listofrms)
        	#listofhk = hsrange[listofrms<minresi+meanerr]
        	#listofrms = [] 
        else:
        	pwvf = 1000.0 * m_w * P_0 * HH / (rho_l * k_0 * T)
        	epwv = 0.3*pwvf
        listofrms = []
        #print(pwvpred)
        for hs in hsrange:
        		HH = hs*1000
        		pwv0 = pwvf
        		xx,yy = model(pwv0,H,T,P,hs)
        		rms, resi = residuals(xx,yy,spwF,spwT,spwE)
        		listofrms.append(rms)
        listofrms = np.array(listofrms) 
        minresi = np.min(listofrms)
        meanerr = (np.median(spwE))/10.0
        #print(hsrange)
        #print(listofrms)
        #print(minresi,meanerr,minresi+meanerr)    
        listofh = hsrange[listofrms<minresi+meanerr]
        listofrms2 = listofrms[listofrms<minresi+meanerr]
        listofh = np.array(listofh)
        listofrms2 = np.array(listofrms2)
        #print(len(listofh))
        #print(listofh)
        #print(listofrms2)
        new_pwv = pwv0
        errorpwv = epwv
        if (len(listofh)>1):
        		finalh, errorh = meanW(listofh,1/listofrms2)
        		#HH = finalh*1000
        		#new_pwv = pwv0
        else:
        		finalh = np.mean(listofh)
        		errorh = 1./(200./5.)
        		#new_pwv = pwv0
        #if bestcurve:
        #	besth = np.mean(hsrange[listofrms==minresi])
        #	deltah = abs(besth-finalh)
        #	finalh = besth
        #	HH = finalh*1000
        #	new_pwv = 1000.0 * m_w * P_0 * HH / (rho_l * k_0 * T)
        #	errorh = errorh+deltah
        xxh,yyh = model(new_pwv,H,T,P,finalh)
        rms, resi = residuals(xx,yy,spwF,spwT,spwE)
        plt.ioff()
        fig = plt.figure()
        plt.ylim(0,0.3)
        if len(spwFFlag) > 0:
             plt.title('Measured: H(%) = '+'{:.1f}'.format(H)+', T(K) = '+'{:.0f}'.format(T)+', P(mb) = '+'{:.0f}'.format(P)+'\n Black: data used Red: data flagged')
        else:
             plt.title('Measured: H(%) = '+'{:.1f}'.format(H)+', T(K) = '+'{:.0f}'.format(T)+', P(mb) = '+'{:.0f}'.format(P))
        plt.plot(xxh,yyh,color='red',label='h(km) obtained = '+'{:.1f}'.format(finalh)+' pm '+'{:.1f}'.format(errorh)+' pwv(mm) obtained = '+'{:.1f}'.format(new_pwv)+' pm '+'{:.1f}'.format(errorpwv))
        #plt.scatter(spwF,spwT,c=abs(resi),vmin=0,vmax=0.03)
        #plt.colorbar()
        plt.errorbar(spwF,spwT,yerr=spwE,color='black',fmt='o')
        if len(spwFFlag) > 0: plt.scatter(spwFFlag,spwTFlag,color='red',marker='o')
        plt.legend()
        plt.xlabel('Frequency (GHz)', fontsize=16)
        plt.ylabel('Opacity', fontsize=16)
        casalog.post('INFO: Measured: H(%) = '+'{:.1f}'.format(H)+', T(K) = '+'{:.0f}'.format(T)+', P(mb) = '+'{:.0f}'.format(P),"INFO")
        casalog.post('INFO: Scale Height h (km) = '+'{:.1f}'.format(finalh)+' pm '+'{:.1f}'.format(errorh),"INFO")
        casalog.post('INFO: Preciptable Water Vapor (mm) = '+'{:.1f}'.format(new_pwv)+' pm '+'{:.1f}'.format(errorpwv),"INFO")
        if not tauPerAnt:
             plt.savefig(msname+'.model.atm.pwv.png')
        else:
             plt.savefig(msname+'.model.atm.pwv.perantenna.png')
        plt.close(fig)
        plt.ion()
        
        return finalh, errorh, new_pwv, errorpwv, rms





    ### get antenna details

    if is_CASA6:
       msmd = casatools.msmetadata()
       msmd.open(msname)
       tb = casatools.table()
       me = casatools.measures()
       qa = casatools.quanta()
    else:
       msmd = casac.msmetadata()



    tb.open(msname+'/ANTENNA')
    antNames=tb.getcol('NAME')
    tb.close()
    lenAnt = len(antNames)
    
    
    ### get full spw details for MS (not only tipping scans)
    tb.open(msname+'/SPECTRAL_WINDOW')
    spwRef=(tb.getcol('REF_FREQUENCY'))    #Channel Zer0
    numCha=(tb.getcol('NUM_CHAN'))
    totalBW=(tb.getcol('TOTAL_BANDWIDTH'))
    spwCntFreq = ((totalBW/numCha)*((numCha-1)/2)+spwRef)   #Channel Width * middle channel to compute center frequency
    #spwCntFreq=np.mean(tb.getcol('CHAN_FREQ'),axis=0)
    tb.close()
    lenSpw = len(spwCntFreq)
    
    
    ### get pointing zenith angle
    # check if pointing sub-table contains data. If not, give error and exit.
    # this should be produced by importasdm in all cases (with_pointing_correction true or false)
    casalog.post('--> Reading antenna pointing data.')
    tb.open(msname+'/POINTING')
    
    # read in elevation vs time, will include data from tips and also pointing if performed
    # Note this is stored in ENCODER in AZELGEO coordinates
    #tb.getcolkeywords('ENCODER')
    #{'MEASINFO': {'Ref': 'AZELGEO', 'type': 'direction'},
    # 'QuantumUnits': array(['rad', 'rad'], dtype='|S4')}
    #
    # CASA coordinate frames:
    # https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/coordinate-frames
    
    # extract pointing data for full observation
    # not super efficient, but majority of data is expected to come from tipping scans.
    # first, get max timestamps per antenna, in case they differ
    maxAntT = 0
    for a in range(lenAnt):
        temptb  = tb.query('ANTENNA_ID=='+str(a))
        lenAntT = len(temptb.getcol('TIME'))
        if lenAntT > maxAntT: maxAntT = lenAntT
    
    if maxAntT == 0:
        casalog.post("*** ERROR: That's strange, the pointing table is empty.","ERROR")
        casalog.post("*** ERROR: Exiting tipopac.","ERROR")
        return
    
    # 0 = time UTC seconds, 1 = zenith angle (deg)
    dataPoint = np.zeros([lenAnt,maxAntT,2])
    me.doframe(me.observatory('VLA'))
    for a in range(lenAnt):
        casalog.post('    processing antenna '+antNames[a]+' ('+str(a+1)+'/'+str(lenAnt)+')')
        temptb                   = tb.query('ANTENNA_ID=='+str(a))		#&& scan?
        lenAntT                  = len(temptb.getcol('TIME'))
        dataPoint[a,0:lenAntT,0] = temptb.getcol('TIME')
        azel                     = temptb.getcol('ENCODER')
        for i in range(lenAntT):
            dataPoint[a,i,1] = 90-np.rad2deg(me.measure(me.direction('AZELGEO',
                                             str(np.rad2deg(azel[0,i]))+'deg',
                                             str(np.rad2deg(azel[1,i]))+'deg'),
                                             'AZEL')['m1']['value'])
    
    tb.close()
    del temptb,azel
 
    msmd.open(msname)    
    ### read in time ranges for tipping scans and get associated spw's
    casalog.post('--> Reading time ranges for tipping scans.')
    # Only the scan with 2 subscans, with intent DO_SKYDIP, are of interest.
    # The others can be ignored, they don't contain any useful data.

    scans     = msmd.scansforintent('*DO_SKYDIP*')
    #scans     = msmd.scansforfield(2)
    lenScans  = len(scans)
    tipSpw    = msmd.spwsforintent('*DO_SKYDIP*')
    #tipSpw    = msmd.spwsforfield(2)

    lenTip = len(tipSpw)
    # get start and end time for each scan
    times    = np.zeros([lenScans,2])       # UTC seconds
    for i in range(lenScans):
        times[i,0] = msmd.timesforscan(scans[i])[0]
        times[i,1] = msmd.timesforscan(scans[i])[-1]
    
    msmd.done()
    

    ### get estimated weighted mean atmospheric temperatures in kinetic temp K
    casalog.post('--> Reading MS surface temperature data and '+\
                     'estimating weighted mean atmospheric temperatures.')
    # sampled every approximately 1 minute
    tb.open(msname+'/WEATHER')
    tmp1 = tb.getcol('TIME')
    # estimate weighted mean atmospheric temperature using
    # Tm ~ 70.2 + 0.72 * Ts
    # from Bevis et al., 1992, J. Geophys. Res. 97(D14), 15,787
    tmp2 = tb.getcol('TEMPERATURE') * 0.72 + 70.2
    tmp2[tb.getcol('TEMPERATURE_FLAG')==1] = np.nan
    tb.close()
    # time (UTC sec), temp (K)
    dataTemp = np.column_stack((tmp1,tmp2))
    del tmp1,tmp2
    
    
    ### read in online flags except for ANTENNA_NOT_ON_SOURCE
    # subreflector errors shouldn't make any difference, but no harm flagging
    if cmdFlag:
        casalog.post('--> Reading online flags.')
        tb.open(msname+'/FLAG_CMD')
        if len(tb.getcol('REASON')) == 0:
            casalog.post("*** ERROR: The online flag table (FLAG_CMD) is empty.","ERROR")
            casalog.post("*** ERROR: Ensure that process_flags=True when running "+\
                         "importasdm.","ERROR")
            casalog.post("*** ERROR: Exiting tipopac.","ERROR")
            return
        
        #temptb     = tb.query("REASON!='ANTENNA_NOT_ON_SOURCE'") ------- Pedro commented
        #dataCmdRaw = temptb.getcol('COMMAND') ------ Pedro Commented and added line below
        dataCmdRaw = tb.query("REASON!='ANTENNA_NOT_ON_SOURCE' and REASON!='SHADOW'and REASON!='CLIP_ZERO_ALL'").getcol('COMMAND')
        lenDataCmd = len(dataCmdRaw)
        # antenna, start time, end time (UTC sec)
        dataCmd    = np.zeros([lenDataCmd,3])
        for f in range(lenDataCmd):
            dataCmd[f,0] = np.where(antNames==dataCmdRaw[f].\
                             replace("'","&&").split('&&')[1])[0][0]
            dataCmd[f,1] = qa.quantity(dataCmdRaw[f].replace("'","&&").\
                             split('&&')[4].split('~')[0],'ymd')['value']*24*3600
            dataCmd[f,2] = qa.quantity(dataCmdRaw[f].replace("'","&&").\
                             split('&&')[4].split('~')[1],'ymd')['value']*24*3600
        
        tb.close()
    
    
    ### read in user-specified flags
    if usrFlag:
        casalog.post('--> Reading user-defined flags.')
        dataUsrRaw = np.loadtxt(flagFile,dtype=str)
        if dataUsrRaw.size == 3:
            dataUsrRaw = dataUsrRaw.reshape([1,3]) 		
        
        lenDataUsr = len(dataUsrRaw)
        # antenna, spw, start time, end time (UTC sec)
        dataUsr    = np.zeros([lenDataUsr,4])
        mintime    = times.min()
        maxtime    = times.max()
        maxf       = lenDataUsr
        f          = 0
        for fraw in range(lenDataUsr):
            allant  = False
            allspw  = False
            myant   = dataUsrRaw[fraw,0].replace("'","").split('=')[1]
            myspw   = dataUsrRaw[fraw,1].replace("'","").split('=')[1]
            mytime1 = qa.quantity(dataUsrRaw[fraw,2].replace("'","").\
                         split('=')[1].split('~')[0],'ymd')['value']*24*3600
            mytime2 = qa.quantity(dataUsrRaw[fraw,2].replace("'","").\
                         split('=')[1].split('~')[1],'ymd')['value']*24*3600
            if myant == '-1':
                allant = True
            else:
                dataUsr[f,0] = np.where(antNames==myant)[0][0]

            
            if myspw == '-1':
                allspw = True
            else:
                dataUsr[f,1] = float(myspw)		
		
            
            dataUsr[f,2] = mytime1
            dataUsr[f,3] = mytime2
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
                dataUsr = np.insert(dataUsr,f,np.zeros([lenSpw-1,4]),axis=0)
                for s in range(lenSpw):
                    dataUsr[f,0] = np.where(antNames==myant)[0][0]
                    dataUsr[f,1] = s
                    dataUsr[f,2] = mytime1
                    dataUsr[f,3] = mytime2
                    f += 1
                
                maxf += lenSpw-1
            elif (allant) and (not allspw):
                # all spws for a given antenna in a given time range
                dataUsr = np.insert(dataUsr,f,np.zeros([lenAnt-1,4]),axis=0)
                for a in range(lenAnt):
                    dataUsr[f,0] = a
                    dataUsr[f,1] = float(myspw)
                    dataUsr[f,2] = mytime1
                    dataUsr[f,3] = mytime2
                    f += 1
                
                maxf += lenAnt-1
            elif (allant) and (allspw):
                # all antennas and all spws in a given time range
                dataUsr = np.insert(dataUsr,f,np.zeros([lenAnt*lenSpw-1,4]),axis=0)
                for a in range(lenAnt):
                    for s in range(lenSpw):
                        dataUsr[f,0] = a
                        dataUsr[f,1] = s
                        dataUsr[f,2] = mytime1
                        dataUsr[f,3] = mytime2
                        f += 1
                
                maxf += lenAnt*lenSpw-1
            else:
                casalog.post("*** ERROR: Manual flags not specified "+\
                             "according to instructions in help.","ERROR")
                casalog.post("*** ERROR: Exiting tipopac.","ERROR")
                return
     
    ### read in switched power psum and pdif per tip scan and apply flags
    casalog.post('--> Reading switched power data.')
    # start by reading in stored Tcals (K)
    tb.open(msname+'/CALDEVICE')
    # antenna, spw, pol
    dataTcalMS = np.zeros([lenAnt,lenSpw,2])
    for a in range(lenAnt):
        for s in range(lenSpw):
            temptb = tb.query('ANTENNA_ID=='+str(a)+'&&SPECTRAL_WINDOW_ID=='+str(s))
            # rows 0/1 = noise tube/solar filter
            # cols 0/1 = R/L
	    # Pedro's comments:
	    # On tipping scan 2020, december 10, there was a missing spw for an antenna
	    # On that case, I added a try. Copy the data form the other spw and let the user know.
            try:
            	dataTcalMS[a,s,0] = temptb.getcol('NOISE_CAL')[0,0,0]
            	dataTcalMS[a,s,1] = temptb.getcol('NOISE_CAL')[0,1,0]
            except:
            	casalog.post('*** WARNING: No data for spw '+str(s)+' at antenna id '+str(a),'WARN')
            	if (s >0):
            		dataTcalMS[a,s,0] = dataTcalMS[a,s-1,0] 
            		dataTcalMS[a,s,1] = dataTcalMS[a,s-1,1]
            		casalog.post('Value used for spw '+str(s))
            	else:
            		casalog.post('Value set as zero.')
		                           
    
    tb.close()
    # create Z caltable to be filled in below
    if caltable:
    	gencaltableZ(msname,caltableZ)
    	# set default values with everything flagged
    	caltableNrows = lenScans * lenSpw * lenAnt
    	tb.open(caltableZ,nomodify=False)
    	tb.addrows(caltableNrows)
    	k = 0
    	for i in range(lenScans):
       	 for a in range(lenAnt):
            for s in range(lenSpw):
                tb.putcell('TIME',              k,(times[i,0]+times[i,1])/2.)
                tb.putcell('FIELD_ID',          k,-1)
                tb.putcell('SPECTRAL_WINDOW_ID',k,s)
                tb.putcell('ANTENNA1',          k,a)
                tb.putcell('ANTENNA2',          k,-1)
                tb.putcell('SCAN_NUMBER',       k,i)
                tb.putcell('FPARAM',            k,np.array([[0.]]))
                tb.putcell('PARAMERR',          k,np.array([[0.]]))
                tb.putcell('FLAG',              k,np.array([[True]],dtype=bool))
                tb.putcell('SNR',               k,np.array([[1.]]))
                # the WEIGHT column can be left with empty cells
                k += 1
    
    	tb.flush()
    	tb.close()
    if calcTcals and caltable:
        tb.open(msname+'/CALDEVICE')
        newtab = tb.copy(caltableT,deep=True,valuecopy=True,norows=True,returnobject=True)
        tb.close()
        newtab.close()
        tb.open(caltableT,nomodify=False)
        tb.addrows(caltableNrows)
        k = 0
        for i in range(lenScans):
            for a in range(lenAnt):
                for s in range(lenSpw):
                    tb.putcell('ANTENNA_ID',        k,a)
                    tb.putcell('SPECTRAL_WINDOW_ID',k,s)
                    tb.putcell('TIME',              k,(times[i,0]+times[i,1])/2.)
                    tb.putcell('NUM_CAL_LOAD',      k,2)
                    tb.putcell('CAL_LOAD_NAMES',    k,np.array([['NOISE_TUBE_LOAD'],['SOLAR_FILTER']]))
                    tb.putcell('NUM_RECEPTOR',      k,2)
                    tb.putcell('NOISE_CAL',         k,np.array([[0.,0.],[0.,0.]]))
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



    #### We will set the same size always
    
    if calcTcals:
        Ndata = 4
    else:
        Ndata = 3

    try:
        data1 = np.zeros([lenScans,lenAnt,lenSpw,2,1200,Ndata])
    except:
        data1 = np.zeros([lenScans,lenAnt,lenSpw,2,200,Ndata])
    
    #### Changes end here
    
    tb.open(msname+'/SYSPOWER')
    for i in range(lenScans):
        casalog.post('--> Gathering data for scan '+str(scans[i])+' ('+str(i+1)+'/'+str(lenScans)+')')
        casalog.filter('WARN')
        msmd.open(msname)
        scanspws = msmd.spwsforscan(scans[i])
        lenScanSpws = len(scanspws)
        msmd.done()
        casalog.filter('INFO')
        for a in range(lenAnt):
            casalog.post('    antenna '+antNames[a]+' ('+str(a+1)+'/'+str(lenAnt)+')')
            for s in scanspws:
                casalog.post('      spw '+str(s)+' ('+str(s-scanspws[0]+1)+'/'+str(lenScanSpws)+')')
                subtb  = tb.query('TIME>='+str(times[i,0])+'&&TIME<='+str(times[i,1])+\
                                  '&&ANTENNA_ID=='+str(a)+'&&SPECTRAL_WINDOW_ID=='+str(s))
                spT    = subtb.getcol('TIME')
                lenSpT = len(spT)
                #print(i,len(spT))
                if lenSpT > 0:
                    # cols 0/1 = R/L
                    pdif  = subtb.getcol('SWITCHED_DIFF')
                    psum  = subtb.getcol('SWITCHED_SUM')		 
                    #rq   = subtb.getcol('REQUANTIZER_GAIN')
                    # rq not needed.  Pdif reported in MS appears to neglect digital gain factor
                    
                    # apply online flags except for ANTENNA_NOT_ON_SOURCE
                    # there is probably a smarter way to do this...
		    # Pedro added if's
                    # cmd case 1: flagging starts and ends within tip
                    if cmdFlag:
                    	tmp = dataCmd[np.where((dataCmd[:,0]==a) &\
                    	                       (dataCmd[:,1]>=spT[0]) &\
                    	                       (dataCmd[:,2]<=spT[-1]))]
                    	for f in range(len(tmp)):
                    	    pdif[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    	    psum[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    
                    # cmd case 2: flagging starts before tip and ends within tip
                    if cmdFlag:
                    	tmp = dataCmd[np.where((dataCmd[:,0]==a) &\
                    	                       (dataCmd[:,1]<=spT[0]) &\
                    	                       (dataCmd[:,2]>=spT[0]) &\
                    	                       (dataCmd[:,2]<=spT[-1]))]
                    	for f in range(len(tmp)):
                    	    pdif[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    	    psum[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    
                    # cmd case 3: flagging starts during tip and ends after tip
                    if cmdFlag:
                    	tmp = dataCmd[np.where((dataCmd[:,0]==a) &\
                    	                       (dataCmd[:,1]>=spT[0]) &\
                    	                       (dataCmd[:,1]<=spT[-1]) &\
                    	                       (dataCmd[:,2]>=spT[-1]))]
                    	for f in range(len(tmp)):
                    	    pdif[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    	    psum[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    
                    # cmd case 4: flagging starts before tip and ends after tip
                    if cmdFlag:
                    	tmp = dataCmd[np.where((dataCmd[:,0]==a) &\
                    	                       (dataCmd[:,1]<=spT[0]) &\
                    	                       (dataCmd[:,2]>=spT[-1]))]
                    	for f in range(len(tmp)):
                    	    pdif[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    	    psum[:,np.where((spT>=tmp[f,1]) & (spT<=tmp[f,2]))] = np.nan
                    
                    # apply manual flags
                    # usr case 1: flagging starts and ends within tip
		    # Pedro added if
                    if usrFlag:
                    	tmp = dataUsr[np.where((dataUsr[:,0]==a) &\
                                           (dataUsr[:,1]==s) &\
                                           (dataUsr[:,2]>=spT[0]) &\
                                           (dataUsr[:,3]<=spT[-1]))]
                    	for f in range(len(tmp)):
                        	pdif[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                        	psum[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                    
                    # usr case 2: flagging starts before tip and ends within tip
		    # Pedro added if
                    if usrFlag:
                    	tmp = dataUsr[np.where((dataUsr[:,0]==a) &\
                                           (dataUsr[:,1]==s) &\
                                           (dataUsr[:,2]<=spT[0]) &\
                                           (dataUsr[:,3]>=spT[0]) &\
                                           (dataUsr[:,3]<=spT[-1]))]
                    	for f in range(len(tmp)):
                        	pdif[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                        	psum[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                    
                    # usr case 3: flagging starts during tip and ends after tip
		    # Pedro added if
                    if usrFlag:
                    	tmp = dataUsr[np.where((dataUsr[:,0]==a) &\
                                           (dataUsr[:,1]==s) &\
                                           (dataUsr[:,2]>=spT[0]) &\
                                           (dataUsr[:,2]<=spT[-1]) &\
                                           (dataUsr[:,3]>=spT[-1]))]
                    	for f in range(len(tmp)):
                        	pdif[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                        	psum[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                    
                    # usr case 4: flagging starts before tip and ends after tip
		    # Pedro added if
                    if usrFlag:
                    	tmp = dataUsr[np.where((dataUsr[:,0]==a) &\
                                           (dataUsr[:,1]==s) &\
                                           (dataUsr[:,2]<=spT[0]) &\
                                           (dataUsr[:,3]>=spT[-1]))]
                    	for f in range(len(tmp)):
                        	pdif[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                        	psum[:,np.where((spT>=tmp[f,2]) & (spT<=tmp[f,3]))] = np.nan
                    
 
                    if usrFlag or cmdFlag: del tmp

                    
                    # put the following info into data1 (for each poln, meh)

	 
                    for x in range(lenSpT):
                        # get pointing zenith angle in deg at spT times
                        # pointing data for the JVLA is recorded every approximately 0.1 sec
                        # don't bother interpolating to swpow timestamps, which for
                        # the JVLA are recorded every 1 sec.  Just take nearest value.

                        data1[i,a,s,:,x,0] = dataPoint[a,(np.abs(dataPoint[a,:,0]-spT[x])).argmin(),1]
			#################################################################################################################### Open file - Save spT[x] and data1 (thinking) Each X save data1 array
			                        
                        # get Tatm in kinetic K at spT times
                        # MS Tsurf is only sampled approximately every minute
                        # So it's perhaps worth interpolating temperatures at the switched power timestamps

                        data1[i,a,s,:,x,1] = np.interp(spT[x],dataTemp[:,0],dataTemp[:,1])
		    	

                    for p in range(2):
                        # calculate Tsys = (Psum/2)/Pdif * Tcal
                        data1[i,a,s,p,0:lenSpT,2] = (psum[p]/2.) / (pdif[p]) * dataTcalMS[a,s,p]
                        #print(i,a,s,p) 
                        #print(len(np.where(data1[i,a,s,p,:,2]>0)[0]),len(np.where(psum>0)),len(np.where(pdif>0)),len(np.where(dataTcalMS[a,s,p]>0))) 
                        # flag Tsys if pdif<0 or psum<0
                        tmpIndx = np.where(pdif[p]<0)[0]
                        #print(tmpIndx)
                        data1[i,a,s,p,tmpIndx,2] = np.nan
                        tmpIndx = np.where(psum[p]<0)[0]
                        #print(tmpIndx)
                        data1[i,a,s,p,tmpIndx,2] = np.nan

                        #uncomments lines below if you wish to save details of Tsys information
                        '''
                        if os.path.exists(msname+'.tsys.info/'):
                             casalog.post("INFO: Plots will be saved at "+msname+".tsys.info","INFO")
                        else:
                             os.system('mkdir '+msname+'.tsys.info')
                             #casalog.post("INFO: Plots will be saved at "+msname+".tsys.info","INFO")
                        Tcal_INFO = open(msname+'.tsys.info/PSUM_PDIFF_'+str(i)+'_ant_'+antNames[a]+'_spw_'+str(s)+'_pol_'+str(p)+'.csv','w')
                        pS = psum[p]
                        pD = pdif[p]
                        for ZA in range(lenSpT):
                             Tcal_INFO.write(str(s)+','+'{:.3f}'.format(spwCntFreq[s])+','+'{:.5f}'.format(pS[ZA])+','+'{:.5f}'.format(pD[ZA])+','+'{:.5f}'.format(dataTcalMS[a,s,p])+','+'{:.5f}'.format(data1[i,a,s,p,ZA,2])+','+'{:.5f}'.format(data1[i,a,s,0,ZA,0])+'\n')
                        Tcal_INFO.close()
                        #casalog.post("INFO: SPW "+str(s)+" Antenna "+antNames[a]+' pol '+str(p)+' N = '+str(len(np.where(data1[i,a,s,p,:,2]>0)[0]))+".","INFO")
                        '''
                        if len(np.where(data1[i,a,s,p,:,2]>0)[0]) < minTipInts:
                            data1[i,a,s,p,:,2] = np.nan
                            if p == 0:
                                polstr = 'R'
                            else:
                                polstr = 'L'
                            
                            casalog.post('*** WARNING: '+antNames[a]+' spw '+str(s)+' poln '+polstr+\
                                         ' completely flagged in scan '+str(scans[i])+' due to insufficient unflagged'+\
                                         ' data after manual flagging or abnormal negative switched power data.','WARN')
  
    tb.close()

    ## proceed with nominated solution type
    casalog.post('--> Calculating opacities and system temperature contributions.')
    if calcTcals:
        casalog.post('    Zenith opacities (tau0), ant+elec contributions (Tae=Tant+Trx1+Trx2), and Tcal_new with % change from Tcal_MS are reported below.')
        casalog.post('    Results will be highlighted if the abs(change) in Tcal_new for R or L is >= '+'{:.0f}'.format(Tdifthresh)+'%.  Check Tcal solutions carefully.')
    else:
        casalog.post('    Zenith opacities (tau0) and ant+elec contributions (Tae=Tant+Trx1+Trx2) are reported below.')
    
    dataTae = np.zeros([lenScans,lenAnt,lenSpw,2])
    if caltable: edataTae = np.zeros([lenScans,lenAnt,lenSpw,2])

    if (not calcTcals) and (tauPerAnt):
        #
        print('OPTION 1: solve for opacity per scan, antenna, and spw (combined solve over both polarizations)')
        #
        if caltable: tb.open(caltableZ,nomodify=False)
        dataopZ = np.zeros([lenScans,lenAnt,lenSpw])
        for i in range(lenScans):
            #casalog.post('--> Processing scan '+str(scans[i])+' ('+str(i+1)+'/'+str(lenScans)+')')
            for a in range(lenAnt):
                #casalog.post('    processing antenna '+antNames[a]+' ('+str(a+1)+'/'+str(lenAnt)+')')
                for s in range(lenSpw):
                    #casalog.post('        processing spectral window '+str(s)+' ('+str(s+1)+'/'+str(lenSpw)+')')
                    # can expect that flagging will be polarization independent
                    # Tsys in data1 can be 0 (dummy value in array) or flagged (NaN)
                    # only process if valid Tsys solutions are available in, say, pol=0
                    indx = np.where(data1[i,a,s,0,:,2]>0)[0]
                    if len(indx)>0:
                        # for this scan, ant, spw: we have 2 datasets (Tsys vs ZA for 2 pols)
                        # and the equations have 3 unknowns (T0_pol1, T0_pol2, tau0)
                        
                        # convert kinetic Twmt to noise temp in K
                        # hmm, to simplify, take mean weighted mean atmospheric temperature during scan
                        Twmtp = k2nt(np.mean(data1[i,a,s,0,indx,1]),spwCntFreq[s])
                        
                        # starting estimate for unknown parameters (T0_pol1, T0_pol2, tau0)
                        se       = [50.,50.,0.2]
			#print(a,s)
			#if (a==11) and (s==27): 
			#	fit = using_curvefit2(data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2],
			#					data1[i,a,s,0,indx,0],Trab,Tuab,Twmtp)
			#	print(fit)
			#print('time to fit')
                        try:

                        		fit, ier = scipy.optimize.leastsq(err_multi_wrap(Twmtp), se,
                                                          			args=(data1[i,a,s,0,indx,0],
                                                          			data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2]))
                        except:
                        		casalog.post('*** WARNING: Not enought unflagged data to fit antenna '+str(a)+\
                        			' at scan '+str(scans[i])+'. Setting opacity = 0 and T=-999 K.','WARN')
                        		NNN=len(se)			
                        		fit=np.zeros(NNN)-999
		
                        #x  = data1[i,a,s,0,indx,0]
                        #y1 = data1[i,a,s,0,indx,2]
                        #y2 = data1[i,a,s,1,indx,2]
                        #from matplotlib import pyplot as plt
                        #plt.plot(x,y1,'b.-',x,func(x,np.r_[fit[0],fit[-1]],Trab,Tuab,Twmtp),'r.-',
                        #         x,y2,'g.-',x,func(x,np.r_[fit[1],fit[-1]],Trab,Tuab,Twmtp),'y.-')
                        #plt.show()
                        
                        dataopZ[i,a,s]   = fit[-1]
                        dataTae[i,a,s,0] = fit[0]-dataTcalMS[a,s,0]/2.
                        dataTae[i,a,s,1] = fit[1]-dataTcalMS[a,s,1]/2.
                        
                        casalog.post('    scan '+str(scans[i])+', '+antNames[a]+', spw '+str(s)+\
                                     ' - tau0: '+'{:.3f}'.format(fit[-1])+', Tae (K): '+\
                                     '{:.2f}'.format(dataTae[i,a,s,0])+' (R), '+\
                                     '{:.2f}'.format(dataTae[i,a,s,1])+' (L)')
                        if caltable:
                        	k = i*lenAnt*lenSpw + a*lenSpw + s
                        	tb.putcell('FPARAM',k,np.array([[fit[-1]]]))
                        	tb.putcell('FLAG',  k,np.array([[False]],dtype=bool))
                        if doPlot: makeplot(antNames[a],s,data1[i,a,s,0,indx,0],data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2],fit[0],fit[1],1.,1.,fit[2],0.02*fit[2],0.05*fit[0],0.05*fit[1],0,0,Twmtp,str(scans[i]))
        if caltable: tb.flush()
        if caltable: tb.close()
    else:
        #
        if (not calcTcals):
        	print('OPTION 2: solve for opacity per scan and spw (combined solve over all antennas and polarizations)')
        else:
                print('OPTION 3: solve for opacity per scan and spw (combined solve over all antennas and polarizations)')
                print('OPTION 3: but also solve for Tcals.')
        #

        dataopZ = np.zeros([lenScans,lenSpw])
        edataopZ = np.zeros([lenScans,lenSpw])

        if calcTcals:
            # 0=R, 1=L, 2=%difference R, 3=%difference L
            # diff = (new-old)/old*100
            dataTcal = np.zeros([lenScans,lenAnt,lenSpw,8])-999
            if caltable: edataTcal = np.zeros([lenScans,lenAnt,lenSpw,4])
     
        for i in range(lenScans):
            casalog.filter('WARN')
            msmd.open(msname)
            scanspws = msmd.spwsforscan(scans[i])
            msmd.done()
            casalog.filter('INFO')
            for s in scanspws:
                # store data in prep for fitting
		# There are 3 layers of bounds to try
                dataZA   = ()
                dataTsys = ()
                boundLower = []
                boundUpper = []
                boundLower2 = []
                boundUpper2 = []
                boundLower3 = []
                boundUpper3 = []
                se = []
                seTcal = []
                AntArr   = []
                antUsed = []
                badTcalRight = []
                badTcalLeft = []
                antStd = []
                badStd = []
                getTruw  = True
                tauTemp = np.zeros(lenAnt)
                stdMin = 1000
                besta = 0
                indxbesta = 0
                for a in range(lenAnt):
                    #indx = np.where(data1[i,a,s,0,:,2]>0)[0]
		    # Pedro changed index
		    # setting fitting boundaries: trUpperLimit - Upper Limit fot Tr (used as a prior criterion either); tauUpperLimit - Upper limit tau, tauLowerLimit - Lower limit tau
		    # setting standard deviation of Tsys as criterion of includ antenna in the spw fitting (stdTsys)
		    # setting limit for standard deviation of the residuals of the first fitting (stdResi)
                    stdResi = 3.
                    trUpperLimit = 300.	
                    if ((spwCntFreq[s]/1e9) > 40):
                    	stdTsys = 20.                    	
                    	if ((spwCntFreq[s]/1e9) > 45):
                    		tauUpperLimit = 0.4
                    		tauLowerLimit = 0.04
                    	else:
                    		tauUpperLimit = 0.3
                    		tauLowerLimit = 0.02
                    else:
                    	if ((spwCntFreq[s]/1e9) > 18):
                    		stdTsys = 15.	
                    	else:
                    		stdTsys = 5.                    	
                    	tauUpperLimit = 0.3
                    	tauLowerLimit = 0.02
                    	#cheking for bad points.	
                    indx = (data1[i,a,s,0,:,2]>0) * (data1[i,a,s,0,:,2]<trUpperLimit)	* (data1[i,a,s,0,:,2] != np.inf) * (data1[i,a,s,0,:,2] != -np.inf)

                    if getTruw: Twmtp   = k2nt(np.mean(data1[i,a,s,0,indx,1]),spwCntFreq[s])

		    # Prior Fitting (pfit, no tcal), checking outliers (output RR and LL correlation), compute std without outliers (noOutRR, noOutLL), find outliers indexs and update indx, recompute std, get first Tau value.
                    if True in indx:	
                    	try:
                    		pfit, ier = scipy.optimize.leastsq(err_multi_wrap(Twmtp),[50.,50.,0.2],args=(data1[i,a,s,0,indx,0],data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2]))
                    		outRR = data1[i,a,s,0,:,2]-(pfit[0] +Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,:,0])))))
                    		outLL = data1[i,a,s,1,:,2]-(pfit[1] +Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,:,0])))))
                    		noOutRR = data1[i,a,s,0,indx,2]-(pfit[0] + Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,indx,0])))))
                    		noOutLL = data1[i,a,s,1,indx,2]-(pfit[1] + Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,indx,0])))))
                    		stdR = np.std(noOutRR)
                    		stdL = np.std(noOutLL)
                    		indx = (data1[i,a,s,0,:,2]>0) * (data1[i,a,s,0,:,2]<trUpperLimit) * (data1[i,a,s,0,:,2] != np.inf) * (data1[i,a,s,0,:,2] != -np.inf) * (abs(outRR[:]) < 2*stdR) * (abs(outLL[:]) < 2*stdL)
                    		pfit, ier = scipy.optimize.leastsq(err_multi_wrap(Twmtp),[50.,50.,0.2],args=(data1[i,a,s,0,indx,0],data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2]))
                    		outRR = data1[i,a,s,0,indx,2]-(pfit[0] + Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,indx,0])))))
                    		outLL = data1[i,a,s,1,indx,2]-(pfit[1] + Twmtp*(1-np.exp(-pfit[-1]/np.cos(np.deg2rad(data1[i,a,s,0,indx,0])))))
                    		stdR = np.std(outRR)
                    		stdL = np.std(outLL)
                    		tauTemp[a] = pfit[-1]
                    	except:
# is it stdR and stdL really necessary?
                    		stdR = 0
                    		stdL = 0
                    		tauTemp[a] = -1
                    if True in indx:

                    	if getTruw:
                            # only need to get Trab,Tuab,Twmtp once, same for all antennas
                            # Unlikely that temperature changes much over 2 mins, so
                            # don't worry about potential flagging differences between ants
                            Twmtp   = k2nt(np.mean(data1[i,a,s,0,indx,1]),spwCntFreq[s])
                            getTruw = False


		      ##################
		     
			# Adding a last criterion prior the last fitting. Checking if ZA is changing properly.
                    	deltaZA = np.max(data1[i,a,s,0,indx,0])-np.min(data1[i,a,s,0,indx,0])
                    	mZA = np.min(data1[i,a,s,0,indx,0])
                    	stdM = stdR+stdL
                    	if stdM < stdMin:
                    	     besta = a
                    	     indxbesta = indx
                    	     stdMin = stdM
			# Select antenna to perform the fit, and adding the boundaries
                    	if (np.max(data1[i,a,s,0,indx,2]) != np.inf) and (np.max(data1[i,a,s,1,indx,2]) != np.inf) and (np.min(data1[i,a,s,0,indx,2]) != -np.inf) and (np.min(data1[i,a,s,1,indx,2]) != -np.inf) and (np.std(data1[i,a,s,0,indx,2])<stdTsys) and (np.std(data1[i,a,s,1,indx,2])<stdTsys) and (deltaZA > 10) and (mZA>30) and (stdR < stdResi) and (stdL < stdResi) and (np.mean(data1[i,a,s,0,indx,2])<trUpperLimit) and (np.mean(data1[i,a,s,1,indx,2])<trUpperLimit):
                    	#if crit01 and crit02 and crit03 and crit04 and crit06:
                    		dataTsys += (data1[i,a,s,0,indx,2],data1[i,a,s,1,indx,2])			        
                    		dataZA   += (data1[i,a,s,0,indx,0],)
                    		boundLower  += [0.,0.8,0.,0.8]	
                    		boundUpper  += [trUpperLimit,1.2,trUpperLimit,1.2]
                    		boundLower2  += [0.,0.7,0.,0.7]	
                    		boundUpper2  += [trUpperLimit,1.2,trUpperLimit,1.2]
                    		boundLower3  += [0.,0.7,0.,0.7]	
                    		boundUpper3  += [trUpperLimit,1.3,trUpperLimit,1.3]
                    		se       += [50.,50.]
                    		AntArr   += [a]
                    		seTcal   += [50.,1.,50.,1.]
                    		antUsed += [antNames[a],]
                    	else:
                    		totalStd = np.std(data1[i,a,s,0,indx,2])+np.std(data1[i,a,s,1,indx,2])
                    		antStd += [a]
                    		badStd += [totalStd]

                if AntArr == []:
                    		dataTsys += (data1[i,besta,s,0,indxbesta,2],data1[i,besta,s,1,indxbesta,2])			        
                    		dataZA   += (data1[i,besta,s,0,indxbesta,0],)
                    		boundLower  += [0.,0.8,0.,0.8]	
                    		boundUpper  += [trUpperLimit,1.2,trUpperLimit,1.2]
                    		boundLower2  += [0.,0.7,0.,0.7]	
                    		boundUpper2  += [trUpperLimit,1.2,trUpperLimit,1.2]
                    		boundLower3  += [0.,0.7,0.,0.7]	
                    		boundUpper3  += [trUpperLimit,1.3,trUpperLimit,1.3]
                    		se       += [50.,50.]
                    		AntArr   += [besta]
                    		seTcal   += [50.,1.,50.,1.]
                    		antUsed += [antNames[besta],]                
                seTcal += [0.2]
                boundLower += [tauLowerLimit]
                boundUpper += [tauUpperLimit]
                boundLower2 += [0.]
                boundUpper2 += [tauUpperLimit]
                boundLower3 += [0.]
                boundUpper3 += [tauUpperLimit]
		#Pedro added Try
                if not calcTcals:
                	try:					
                		fit, ier = scipy.optimize.leastsq(err_multi_wrap(Twmtp),se,args=dataZA+dataTsys)
                	except:		
                		casalog.post('*** WARNING: Not enought unflagged data to fit spw '+str(s)+' at scan '+str(scans[i])+\
                			'. Setting opacity = 0 and T=-999 K.','WARN')
                		NNN=len(se)			
                		fit=np.zeros(NNN)-999
                	if fit[-1] < 0: 
                		fit[-1] = np.mean(tauTemp[tauTemp>0])
                		efit[-1] = 3*np.std(tauTemp[tauTemp>0])
                		casalog.post('Scan '+str(scans[i])+' opacity for spw '+str(s)+' at '+'{:.3f}'.format(spwCntFreq[s]/1e9)+' GHz computed per antenna without fitting Tcal.','WARN')
                		print('Scan '+str(scans[i])+' opacity for spw '+str(s)+' at '+'{:.3f}'.format(spwCntFreq[s]/1e9)+' GHz computed per antenna without fitting Tcal.')
                		anyBadRight = False
                		anyBadLeft = False
                	else:
                		anyBadRight = False
                		anyBadLeft = False
                		if doPlot:
                			inan = 0
                			inza = 0
                			efit = 0.3*fit
                			tau = fit[-1]
                			etau = 0.3*tau
                			for a in AntArr:
                				makeplot(antNames[a],s,dataZA[inza],dataTsys[inan],dataTsys[inan+1],fit[inan],fit[inan+1],1.,1.,tau,etau,efit[inan],efit[inan+1],0,0,Twmtp,str(scans[i]))
                				inan += 2
                				inza += 1
                else:
                	fit, Tcal, efit, eTcal, ErrorAnt, VersionFit  = fitting_Tcal(dataZA+dataTsys,seTcal,boundLower,boundUpper,boundLower2,boundUpper2,boundLower3,boundUpper3,Twmtp)
                	if fit[-1] < 0:
                		#print(tauTemp)
                		fit[-1] = np.mean(tauTemp[tauTemp>0])
                		efit[-1] = 3*np.std(tauTemp[tauTemp>0])
                		casalog.post('After fit attempt: '+str(VersionFit)+ ' Scan '+str(scans[i])+' opacity for spw '+str(s)+' at '+'{:.3f}'.format(spwCntFreq[s]/1e9)+' GHz computed per antenna without fitting Tcal.','WARN')
                		print('Scan '+str(scans[i])+' opacity for spw '+str(s)+' at '+'{:.3f}'.format(spwCntFreq[s]/1e9)+' GHz computed per antenna without fitting Tcal.')
                		anyBadRight = False
                		anyBadLeft = False
                	else:
                		inan = 0
                		inza = 0
                		tau = fit[-1]
                		etau = efit[-1]
                		anyBadRight = False
                		anyBadLeft = False
                		for a in AntArr:
                			#if doPlot and (not ErrorAnt[inan] or not ErrorAnt[inan+1]): makeplot(antNames[a],s,dataZA[inza],dataTsys[inan],dataTsys[inan+1],fit[inan],fit[inan+1],Tcal[inan],Tcal[inan+1],tau,Twmtp)
                			if doPlot: makeplot(antNames[a],s,dataZA[inza],dataTsys[inan],dataTsys[inan+1],fit[inan],fit[inan+1],Tcal[inan],Tcal[inan+1],tau,etau,efit[inan],efit[inan+1],eTcal[inan],eTcal[inan+1],Twmtp,str(scans[i]))
                			if not ErrorAnt[inan]:
                				dataTcal[i,a,s,0]   = Tcal[inan]*dataTcalMS[a,s,0]
                				dataTcal[i,a,s,2] = (Tcal[inan]-1)*100.
                				dataTcal[i,a,s,4] = Tcal[inan]
                				dataTcal[i,a,s,6] = dataTcalMS[a,s,0]
                				if caltable: 
                					edataTcal[i,a,s,0] = eTcal[inan]*dataTcalMS[a,s,0]
                					edataTcal[i,a,s,2] = abs((Tcal[inan]+eTcal[inan]-1)*100.-(Tcal[inan]-1)*100.)
                			else:
                				badTcalRight += [antNames[a],]
                				anyBadRight = True
                			if not ErrorAnt[inan+1]:				
                				dataTcal[i,a,s,1]   = Tcal[inan+1] * dataTcalMS[a,s,1]
                				dataTcal[i,a,s,3] = (Tcal[inan+1]-1)*100.
                				dataTcal[i,a,s,5] = Tcal[inan+1]
                				dataTcal[i,a,s,7] = dataTcalMS[a,s,1]
                				if caltable:
                					edataTcal[i,a,s,1] = eTcal[inan+1] * dataTcalMS[a,s,1]
                					edataTcal[i,a,s,3] = abs((Tcal[inan+1]+eTcal[inan+1]-1)*100.-(Tcal[inan+1]-1)*100.)
                			else:
                				badTcalLeft += [antNames[a],]
                				anyBadLeft = True
                			inan += 2
                			inza += 1

                	casalog.post('Scan '+str(scans[i])+': Antennas used for spw '+str(s)+' to get tau:'+str(antUsed),'INFO')
                	if anyBadRight and caltable: casalog.post('Scan '+str(scans[i])+': The following antennas for spw '+str(s)+' do not show a good Tcal solution for R polarization:'+str(badTcalRight),'INFO')
                	if anyBadLeft and caltable: casalog.post('Scan '+str(scans[i])+': The following antennas for spw '+str(s)+' do not show a good Tcal solution for L polarization:'+str(badTcalLeft),'INFO')
                	casalog.post('Fit attempt: '+str(VersionFit)+'. Scan '+str(scans[i])+' opacity for spw '+str(s)+' at '+'{:.3f}'.format(spwCntFreq[s]/1e9)+' GHz: '+'{:.3f}'.format(fit[-1])+' pm '+'{:.3f}'.format(efit[-1]),'INFO')
			

                #import matplotlib.pyplot as plt
                #a=4
                #p=0; plt.plot(dataZA[a],dataTsys[2*a+p],'b.-',dataZA[a],func(dataZA[a],[fit[2*a+p],fit[-1]],Trab,Tuab,Twmtp),'r-')
                #p=1; plt.plot(dataZA[a],dataTsys[2*a+p],'b.-',dataZA[a],func(dataZA[a],[fit[2*a+p],fit[-1]],Trab,Tuab,Twmtp),'r-')
                dataopZ[i,s] = fit[-1]
                edataopZ[i,s] = efit[-1]
                m = 0
                for a in AntArr:
                    logpriority = 'INFO'
                    if calcTcals:
			#NN = np.isnan(dataTcal[i,a,s,0])
                        if not ErrorAnt[m]:
                        	dataTae[i,a,s,0] = fit[m] #   - dataTcal[i,a,s,0]/2.
                        	if caltable: edataTae[i,a,s,0] = efit[m]
                        else:
                        	dataTae[i,a,s,0] = -999
                        	if caltable: edataTae[i,a,s,0] = -999
			#NN = np.isnan(dataTcal[i,a,s,1])
                        if not ErrorAnt[m+1]:
                        	dataTae[i,a,s,1] = fit[m+1] # - dataTcal[i,a,s,1]/2.
                        	if caltable: edataTae[i,a,s,1] = efit[m+1]
                        else:
                        	dataTae[i,a,s,1] = -999
                        	if caltable: edataTae[i,a,s,1] = -999
                        extrastr = ', Tcal_new (K): '+\
                                   '{:.2f}'.format(dataTcal[i,a,s,0])+' (R), '+\
                                   '{:.2f}'.format(dataTcal[i,a,s,1])+' (L), '+\
                                   '% change from Tcal_MS: '+\
                                   '{:.1f}'.format(dataTcal[i,a,s,2])+' (R), '+\
                                   '{:.1f}'.format(dataTcal[i,a,s,3])+' (L)'
                        if (np.abs(dataTcal[i,a,s,2])>=Tdifthresh) or (np.abs(dataTcal[i,a,s,3])>=Tdifthresh):
                            logpriority = 'WARN'
                    else:
                        dataTae[i,a,s,0] = fit[2*m] #  - dataTcalMS[a,s,0]/2.
                        dataTae[i,a,s,1] = fit[2*m+1] # - dataTcalMS[a,s,1]/2.
                        if caltable:
                        	edataTae[i,a,s,0] = efit[2*m]
                        	edataTae[i,a,s,1] = efit[2*m+1]
                        extrastr = ''
                    #casalog.post('    scan '+str(scans[i])+', spw '+str(s)+', '+antNames[a]+\
                    #             ' - tau0: '+'{:.3f}'.format(fit[-1])+', Tae (K): '+\
                    #             '{:.2f}'.format(dataTae[i,a,s,0])+' (R), '+\
                    #             '{:.2f}'.format(dataTae[i,a,s,1])+' (L)'+extrastr,logpriority)
                    m += 2
                    if caltable:
                    	 k  = i*lenAnt*lenSpw + a*lenSpw + s
                    	 tb.open(caltableZ,nomodify=False)
                    	 tb.putcell('FPARAM',int(k),np.array([[fit[-1]]]))
                    	 tb.putcell('FLAG',  int(k),np.array([[False]],dtype=bool))
                    	 tb.flush()
                    	 tb.close()	
                    	 if calcTcals:
                    	 	tb.open(caltableT,nomodify=False)
                    	 	tb.putcell('NOISE_CAL',int(k),np.array([[dataTcal[i,a,s,0],dataTcal[i,a,s,1]],[0.,0.]]))
                    	 	tb.flush()
                    	 	tb.close()
			  
    
    # print out summary statistics
    casalog.post('--> Print summary statistics for zenith opacity (over antenna) in nepers: ')
    if (not calcTcals):
        if tauPerAnt:
            #
            # OPTION 1: opacity was solved per scan, antenna, and spw
            #
            casalog.post('    median, median absolute deviation, min outlier, max outlier.')
            for i in range(lenScans):
                casalog.post('    scan '+str(scans[i])+':')
                casalog.filter('WARN')
                msmd.open(msname)
                scanspws = msmd.spwsforscan(scans[i])
                msmd.done()
                casalog.filter('INFO')
                for s in scanspws:
                    # value of zero could be present if all data was flagged
                    # don't let this contribute to statistics
                    sA = dataopZ[i,:,s]
                    sB = sA[np.abs(dataopZ[i,:,s])>0]
		    #if sB==[]:
			#	sB=0
		    
		    #Pedro added the if
                    if (len(sB)!=0):
                    	s1 = np.median(sB)
                    	s2 = np.median(np.abs(sB-s1))
                    	s3 = np.min(sB)
                    	s4 = np.max(sB)
                    	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+\
                                 '{:6.3f}'.format(s1)+', '+'{:6.3f}'.format(s2)+', '+\
                                 '{:6.3f}'.format(s3)+', '+'{:6.3f}'.format(s4))
                    else:
                    	casalog.post('*** WARNING: Not enought unflagged data to fit spw '+str(s)+'.','WARN')
            msmd.open(msname)
            scanspws = msmd.spwsforintent('*DO_SKYDIP*')
            msmd.done()
			
        else:
            #
            # OPTION 2: opacity was solved per scan and spw
            #
            #for i in range(lenScans):
            #    casalog.post('    scan '+str(scans[i])+':')
            #    casalog.filter('WARN')
            #    msmd.open(msname)
            #    scanspws = msmd.spwsforscan(scans[i])
            #    msmd.done()
            #    casalog.filter('INFO')
            msmd.open(msname)
            scanspws = msmd.spwsforintent('*DO_SKYDIP*')
            msmd.done()
            casalog.post('Printing opacities per spw:')
            for s in scanspws:
            	Tau = dataopZ[:,s]
            	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+'{:6.3f}'.format(np.median(Tau[Tau>0])))
    else:
        #
        # OPTION 3: same as option 2
        #
        #for i in range(lenScans):
        #    casalog.post('    scan '+str(scans[i])+':')
        #    casalog.filter('WARN')
        #    msmd.open(msname)
        #    scanspws = msmd.spwsforscan(scans[i])
        #    msmd.done()
        #    casalog.filter('INFO')
        msmd.open(msname)
        scanspws = msmd.spwsforintent('*DO_SKYDIP*')
        msmd.done()
        casalog.post('Printing opacities per spw:')
        for s in scanspws:
            	Tau = dataopZ[:,s]
            	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+'{:6.3f}'.format(np.median(np.median(Tau[Tau>0]))))

    if caltable:
    	casalog.post('--> Print summary statistics for Tae (over antenna and polarization) in K: ')
    	casalog.post('    median, median absolute deviation, min outlier, max outlier')
    	for i in range(lenScans):
        	casalog.post('    scan '+str(scans[i])+':')
        	casalog.filter('WARN')
        	msmd.open(msname)
        	scanspws02 = msmd.spwsforscan(scans[i])
        	msmd.done()
        	casalog.filter('INFO')
        	for s in scanspws02:
        	    # value of zero could be present if all data was flagged
        	    # don't let this contribute to statistics
		    # Pedro added next if
        	    sA = dataTae[i,:,s,:]
        	    sB = sA[np.abs(dataTae[i,:,s,:])>0] 
        	    #sB = sA[(dataTae[i,:,s,:])>0 * (dataTae[i,:,s,:]) < 600]
        	    sindex = (sB>0.) * (sB<600.)
        	    if (len(sB[sindex])!=0):
        	    	s1 = np.median(sB[sindex])
        	    	s2 = np.median(np.abs(sB[sindex]-s1))
        	    	s3 = np.min(sB[sindex])
        	    	s4 = np.max(sB[sindex])
        	    	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+\
        	                 '{:8.3f}'.format(s1)+', '+'{:8.3f}'.format(s2)+', '+\
        	                 '{:8.3f}'.format(s3)+', '+'{:8.3f}'.format(s4))
        	    else:
        	    	casalog.post('*** WARNING: Not enought unflagged data to fit spw '+str(s)+'.','WARN')

  
    if calcTcals and caltable:
        casalog.post('--> Print summary statistics for new Tcal solutions (over antenna and polarization) in K: ')
        casalog.post('    median, median absolute deviation, min outlier, max outlier')
        casalog.post('    scan '+str(scans[i])+':')
        casalog.filter('WARN')
        msmd.open(msname)
        scanspws02 = msmd.spwsforscan(scans[i])
        msmd.done()
        casalog.filter('INFO')
        for s in scanspws02:
            # value of zero could be present if all data was flagged
            # don't let this contribute to statistics
	    #Pedro added NN and next if
            sA = dataTcal[i,:,s,0:2]
            sB = sA[np.abs(dataTcal[i,:,s,0:2])>0. * (dataTcal[i,:,s,0:2] < 100.)]
            if (len(sB)!=0):
            	s1 = np.median(sB)
            	s2 = np.median(np.abs(sB-s1))
            	s3 = np.min(sB)
            	s4 = np.max(sB)
            	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+'{:8.3f}'.format(s1)+', '+'{:8.3f}'.format(s2)+', '+'{:8.3f}'.format(s3)+', '+'{:8.3f}'.format(s4))
            else:
            	casalog.post('*** WARNING: Not possible to determine any possible Tr for spw '+str(s)+'.','WARN')


        casalog.post('--> Print summary statistics for dTcal(%) = (Tcal_new-Tcal_ref)/Tcal_ref*100 (over antenna and polarization): ')
        casalog.post('    median, median absolute deviation, min outlier, max outlier')
        casalog.post('    scan '+str(scans[i])+':')
        casalog.filter('WARN')
        msmd.open(msname)
        scanspws02 = msmd.spwsforscan(scans[i])
        msmd.done()
        casalog.filter('INFO')
        for s in scanspws02:
            # value of zero could be present if all data was flagged
            # don't let this contribute to statistics
	    #Pedro added NN and next if
            sA = dataTcal[i,:,s,2:4]
	    #NN = np.isnan(dataTcal).any()
	    #if (NN==False):
            s0 = sA[np.abs(dataTcal[i,:,s,0:2])>0.]	
            sB = sA[np.abs(dataTcal[i,:,s,2:4])>0.]
            sindex = (sB>(-19.999)) * (sB<19.999) * (s0 != 0.0)
            if (len(sB[sindex])!=0):
            	s1 = np.median(sB[sindex])
            	s2 = np.median(np.abs(sB[sindex]-s1))
            	s3 = np.min(sB[sindex])
            	s4 = np.max(sB[sindex])
            	casalog.post('      spw '+str(s)+' ({:.4f} GHz): '.format(spwCntFreq[s]/1e9)+'{:8.3f}'.format(s1)+', '+'{:8.3f}'.format(s2)+', '+'{:8.3f}'.format(s3)+', '+'{:8.3f}'.format(s4))
            else:
            	casalog.post('*** WARNING: Not possible do get ant resonalbe Tcal for spw '+str(s)+'.','WARN')

    if doModel:
        casalog.post("INFO: Starting atm model.","INFO")
        alltau = np.zeros(len(spwCntFreq))
        ealltau = np.zeros(len(spwCntFreq))
        inds = 0
        tb.open(msname+'/WEATHER')
        Temperature = np.median(tb.getcol('TEMPERATURE'))
        Humidity = np.median(tb.getcol('REL_HUMIDITY'))
        Pressure = np.median(tb.getcol('PRESSURE'))
        Dew_Point = np.median(tb.getcol('DEW_POINT'))
        tb.close()
        tipOpacity = np.zeros(lenSpw)
        tipFreq = np.zeros(lenSpw)
        tipError = np.zeros(lenSpw)
        tipErrorModel = np.zeros(lenSpw)
        inds = 0
        for s in scanspws:
    	      if not tauPerAnt: 
                  Tau = dataopZ[:,s]
                  eTau = edataopZ[:,s]
    	      else:
                  Tau = dataopZ[:,:,s]
                  emax = np.max(Tau[Tau>0])-np.median(Tau[Tau>0])
                  emin = np.median(Tau[Tau>0])-np.min(Tau[Tau>0])
                  if emax > emin:
                      eTau = emax/20
                  else:
                      eTau = emin/20 
    	      tipOpacity[inds] = np.median(Tau[Tau>0])
    	      tipFreq[inds] = spwCntFreq[s]
    	      if not tauPerAnt:
    	           tipError[inds] = np.median(eTau[Tau>0])
    	      else:
    	           tipError[inds] = eTau
    	      inds += 1
        hscale, errfit, pwv, errpwv, rmsTau = fitATM(tipFreq,tipOpacity,tipError,Temperature,Pressure,Humidity,Dew_Point)
        inds = 0
        for si in range(lenSpw):
              FGHz = tipFreq[si]/1e9
              freq_opacity = estimateOpacity(pwvmean=pwv,reffreq=FGHz,altitude=2124,P=Pressure,H=Humidity,T=Temperature,h0=hscale,maxAltitude=20.0,verbose=False)
              tipErrorModel[si] = abs(freq_opacity[0]-tipOpacity[si])
        typical_error = np.max(tipErrorModel)       
        for si in range(len(spwCntFreq)):
              FGHz = spwCntFreq[si]/1e9
              freq_opacity = estimateOpacity(pwvmean=pwv,reffreq=FGHz,altitude=2124,P=Pressure,H=Humidity,T=Temperature,h0=hscale,maxAltitude=20.0,verbose=False)
              alltau[si] = freq_opacity[0]
              ealltau[si] = typical_error
        if not caltable:
            return alltau
        else:            
    	    if not tauPerAnt:
                newdataopZ = np.zeros([lenScans,lenSpw])
                enewdataopZ = np.zeros([lenScans,lenSpw])
                for i in range(lenScans):
                     for ii in range(lenSpw):
                             if dataopZ[i,ii] > 0: 
                                           newdataopZ[i,ii] = alltau[ii]
                                           enewdataopZ[i,ii] = ealltau[ii]
                return spwCntFreq, antNames, newdataopZ, dataTae, dataTcal, enewdataopZ, edataTae, edataTcal, hscale, errfit, pwv, errpwv, rmsTau
    	    else:
                newdataopZ = np.zeros([lenScans,lenAnt,lenSpw])
                enewdataopZ = np.zeros([lenScans,lenAnt,lenSpw])
                for i in range(lenScans):
                     for ii in range(lenSpw):
                             for an in range(lenAnt):
                                   if dataopZ[i,an,ii] > 0: 
                                                     newdataopZ[i,an,ii] = alltau[ii]
                                                     enewdataopZ[i,an,ii] = ealltau[ii]
                edataTae = 0.1*dataTae
                return spwCntFreq, antNames,newdataopZ, dataTae, enewdataopZ, edataTae, hscale, errfit, pwv, errpwv, rmsTau 
    else:
       if not caltable:
    	    if tauPerAnt:
    	         finalTau = np.zeros([len(antNames),lenSpw])
    	    else:
    	         finalTau = np.zeros(lenSpw)
    	    for s in range(lenSpw):
    	        if tauPerAnt:
                     for an in range(lenAnt):
                        Tau = dataopZ[:,an,s]
                        finalTau[an,s] = np.median(Tau[Tau>0])
    	        else:
    	             Tau = dataopZ[:,s]
    	             finalTau[s] = np.median(Tau[Tau>0])
    	    if tauPerAnt:
                 return antNames, finalTau
    	    else:
    	         return finalTau
       else:
    	    if not tauPerAnt:
    	    	return spwCntFreq, antNames, dataopZ, dataTae, dataTcal, edataopZ, edataTae, edataTcal
    	    else:
    	    	return spwCntFreq, antNames, dataopZ, dataTae



