#print("Hello World!")
import numpy as np
from scipy.special import sph_harm
from scipy.optimize import fsolve
# importing libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
# Example: Plotting a triangle and a circle in 3D
#import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


global plx, pltt, posx, velx, accx
# Constant acceleration computation from final velocity v
def Constant_Acceleration_Velocity(xb, xd, u, v, a, start, end, incr, tstart, plx, pltt, posx, velx, accx):
    t1 = (v - u) / a
    x1 = u * t1 + 0.5 * a * t1**2
    for i_count in range(start, end, incr):
        tinc = (i_count-start) * (t1 / (end-start-1))
        plx[i_count] = xb + u * tinc + 0.5 * a * tinc**2
        pltt[i_count] = tstart + tinc
        posx[i_count] = plx[i_count]
        velx[i_count] = u + a*tinc
        accx[i_count] = a
    xb = xb + x1 
    print("Constant_Acceleration", "xb =", xb, "t1", t1)
    return xb, t1 

# Constant velocity computation
def Constant_Velocity(xb, xdd, u,  start, end, incr, tstart, plx, pltt, posx, velx, accx):
    t4 = xdd/ u
    for i_count in range(start, end, incr):
        tinc = (i_count-start) * (t4 / (end-start-1))
        plx[i_count] = xb + u * tinc
        pltt[i_count] = tstart + tinc
        posx[i_count] = plx[i_count]
        velx[i_count] = u
        accx[i_count] = 0
    xb = xb + xdd
    print("Constant_Velocity", "xb =", xb, "t4", t4)
    return xb, t4 
#CA - Constant Acceleration CV - Constant Velocity CA - Constant Acceleration
def CA_CV_CA(str,accn1,vel11,vel12,vel2,cdis2,accn3,vel31,vel32,xb,xd,vbx,vdx,S_min,S_max,
              plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1 = Constant_Acceleration_Velocity(xb, S_min, vel11, vel12, accn1,  0, 10, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt2  = Constant_Velocity(xb, cdis2, vel2,  10, 20, 1, tt1, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  20, 30, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    return xb 
#CA - Constant Acceleration  CA - Constant Acceleration
def CA_CA(str,accn1,vel11,vel12,accn3,vel31,vel32,xb,xd,vbx,vdx,S_min,S_max, 
          plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Acceleration_Velocity(xb, S_min, vel11, vel12, accn1,  0, 16, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  15, 30, 1, tt1, plx, pltt, posx, velx, accx)
    return xb 
#CA - Constant Acceleration  CV - Constant Velocity CA - Constant Acceleration  CA - Constant Acceleration
def CV_CA_CA_CA(str,cdis1,vel11,accn2,vel21,vel22,accn3,vel31,vel32,accn4,vel41,vel42,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Velocity(xb, cdis1, vel11,  0, 8, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt2  = Constant_Acceleration_Velocity(xb, S_min, vel21, vel22, accn2,  7, 16, 1, tt1, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  15, 23, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    xb, tt4  = Constant_Acceleration_Velocity(xb, xd, vel41, vel42, accn4,  22, 30, 1, tt2+tt1+tt3, plx, pltt, posx, velx, accx)
    return xb 
#CA - Constant Acceleration  CV - Constant Velocity CA - Constant Acceleration  CA - Constant Acceleration
def CA_CV_CA_CA(str,accn1,vel11,vel12,cdis2,vel21,accn3,vel31,vel32,accn4,vel41,vel42,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Acceleration_Velocity(xb, S_min, vel11, vel12, accn1,  0, 7, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt2  = Constant_Velocity(xb, cdis2, vel21,  7, 15, 1, tt1, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  15, 22, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    xb, tt4  = Constant_Acceleration_Velocity(xb, xd, vel41, vel42, accn4,  22, 30, 1, tt2+tt1+tt3, plx, pltt, posx, velx, accx)
    return xb 
#CA - Constant Acceleration  CA - Constant Acceleration CV - Constant Velocity CA - Constant Acceleration
def CA_CA_CV_CA(str,accn1,vel11,vel12,accn2,vel21,vel22,cdis3,vel31,accn4,vel41,vel42,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Acceleration_Velocity(xb, S_min, vel11, vel12, accn1,  0, 8, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel21, vel22, accn2, 7, 16, 1, tt1, plx, pltt, posx, velx, accx )
    xb, tt2  = Constant_Velocity(xb, cdis3, vel31,  15, 23, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    xb, tt4  = Constant_Acceleration_Velocity(xb, xd, vel41, vel42, accn4,  22, 30, 1, tt2+tt1+tt3, plx, pltt, posx, velx, accx)
    return xb 
#CV - Constant Velocity CA - Constant Acceleration
def CV_CA(str,vel1,cdis1,accn2,vel21,vel22,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Velocity(xb, cdis1, vel1,  0, 10, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel21, vel22, accn2,  10, 30, 1, tt1, plx, pltt, posx, velx, accx)
    return xb 
#CV - Constant Velocity CA - Constant Acceleration  CA - Constant Acceleration
def CV_CA_CA(str,cdis1,vel11,accn2,vel21,vel22,accn3,vel31,vel32,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Velocity(xb, cdis1, vel11,  0, 11, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt2  = Constant_Acceleration_Velocity(xb, xd, vel21, vel22, accn2,  10, 21, 1, tt1, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  20, 30, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    return xb 
#CA - Constant Acceleration  CA - Constant Acceleration  CA - Constant Acceleration
def CA_CA_CA(str,accn1,vel11,vel12,accn2,vel21,vel22,accn3,vel31,vel32,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx):
    print(str)
    print("xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
    xb, tt1  = Constant_Acceleration_Velocity(xb, S_min, vel11, vel12, accn1,  0, 11, 1, 0, plx, pltt, posx, velx, accx)
    xb, tt2  = Constant_Acceleration_Velocity(xb, xd, vel21, vel22, accn2,  10, 21, 1, tt1, plx, pltt, posx, velx, accx)
    xb, tt3  = Constant_Acceleration_Velocity(xb, xd, vel31, vel32, accn3,  20, 30, 1, tt2+tt1, plx, pltt, posx, velx, accx)
    return xb 
   ####################################################

        # if vdx>vbx:
        #     if vbx>=0:
        #         if np.abs(xd-xb) <= S_min:
        #         elif (xb-xd)<=S_max:
        #         else:#  (xb-xd)>S_max
        #     else: # vbx<0
        #         if np.abs(xd-xb) <= S_min:
        #         elif (xb-xd)<=S_max:
        #         else:#  (xb-xd)>S_max
        # else: # vdx<vbx
        #     if vbx>=0:
        #         if np.abs(xd-xb) <= S_min:
        #         elif (xb-xd)<=S_max:
        #         else:#  (xb-xd)>S_max
        #     else: # vbx<0
        #         if np.abs(xd-xb) <= S_min:
        #         elif (xb-xd)<=S_max:
        #         else:#  (xb-xd)>S_max

####################################################
def Trajectory(xb,xd,vbx,vdx,a_max,a_min,vmax,vmin, plx, pltt, posx, velx, accx):
    global S_min,S_max
    print("hello")
    estop = 0
    a_max2 = 2.0 * a_max
    a_min2 = 2.0 * a_min
    if vbx <= vdx:
        vdx2mvbx2 = vdx**2 - vbx**2
        vmax2mvbx2 = vmax**2 - vbx**2
        S_min = np.abs(vdx2mvbx2 / a_max2)
        S_max = np.abs((vmax2mvbx2 / a_max2) + ((vmax**2 - vdx**2) / a_min2))

    else:  # vbx > vdx
        vdx2mvbx2 = vdx**2 - vbx**2
        vmax2mvbx2 = vmax**2 - vdx**2
        S_min = np.abs(vdx2mvbx2 / a_min2)
        print("S_min:", S_min)
        S_max = np.abs((vmax2mvbx2 / a_max2) + ((vmax**2 - vdx**2) / a_min2))
    print("vbx =", vbx, "; vdx = ", vdx,"; xb = ", xb, "; xd = ", xd)
    print("S_min:", S_min, "S_max:", S_max)


    if xd>=xb:
        if np.abs(xd-xb) <= S_min:
            if vbx>=0:
                if vdx>vbx:
    # Section 1.1.1.1
                    str1 = "here - 1: xd>xb & vdx > vbx & vbx>0 & (xd-xb)<S_min "
                    xb = CA_CV_CA(str1,a_min,vbx,-vbx,-vbx,
                                            -(S_min-(xd-xb)),a_max,-vbx,vdx,xb,xd,vbx,vdx,S_min,S_max,
                                            plx, pltt, posx, velx, accx) 
                else:
    #Section 1.2.1.1
                    xbb = xb
                    cdis = (S_min+(xd-xb))
                    str1 = "here 1b: xd>xb & vdx < vbx & vbx>0 & (xd-xb)<S_min vdx<0"
                    xb = CV_CA(str1,vbx,cdis,a_min,vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                    if (np.abs(xd-xb)>0.01):
                        xb=xbb
    # part of Section 1.2.1.1
                        str1 = "here 1a: xd>xb & vdx < vbx & vbx>0 & (xd-xb)<S_min & vdx>0"
                        cdis = -(S_min-(xd-xb))
                        xb  = CA_CV_CA_CA(str1,a_min,vbx,-vbx,cdis,(-vbx if cdis < 0 else vdx),
                                        a_max,-vbx,vbx,a_min,vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)   
                    #end vdx vbx loop
            else: #vbx<0
                if vdx>vbx:
    #Section 1.1.2.1
                    str1 = "here 2:xd>xb & vdx > vbx & vbx<0 & (xd-xb)<S_min "
                    cdis = -(S_min-(xd-xb))
                    xb  = CV_CA_CA(str1,cdis,vbx,a_max,vbx,-vbx,
                                            (a_min if vdx<=-vbx else a_max),-vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                else: # vdx<vbx
    # Section 1.2.2.1
                    str1 = "here 2b: xd>xb & vdx < vbx & vbx<0 & S_min>(xd-xb) "
                    xb  = CV_CA_CA_CA(str1,-(S_min-(xd-xb)),vbx,a_max,vbx,-vbx,a_max,-vbx,-vdx,
                                a_min,-vdx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
            #end vbx loop
        elif xd-xb<=S_max:
            if vbx>=0:
                if vdx>vbx:
    # Section 1.1.1.2
                    print("xd>xb &","vdx > vbx &"," vbx>0 & ","S_min<(xd-xb)<s_max")
                    print("here 3","xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
                    VV = np.sqrt((2 * a_max * a_min * (xd - xb) + a_min * vbx**2 - a_max * vdx**2) / (a_min - a_max))
                    str1 = "here 3:  xd>xb & vdx > vbx & vbx>0 & S_min<(xd-xb)<s_max"
                    xb  = CA_CA(str1,a_max,vbx,VV,a_min,VV,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                else:
    # Section 1.2.1.2
                    VV = np.sqrt((2 * a_max * a_min * (xd - xb) + a_min * vbx**2 - a_max * vdx**2) / (a_min - a_max))
                    str1 = "here 3a:  xd>xb & vdx < vbx & vbx>0 & S_min<(xd-xb)<S_max"
                    xb  = CA_CA(str1,a_max,vbx,VV,a_min,VV,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                #end vdx vbx loop
            else: #vbx<0
    # Section 1.2.2.2 ans 1.1.2.2
                VV = np.sqrt((2 * a_max * a_min * (xd - xb) + a_min * vbx**2 - a_max * vdx**2) / (a_min - a_max))
                str1 = "here 3a:  xd>xb & vdx <= vbx & vbx<0 & S_min<(xd-xb)<S_max"
                xb  = CA_CA(str1,a_max,vbx,VV,a_min,VV,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
            #end vbx loop
        else: #(xd-xb)>S_max
    # Sections 1.1.1.3; 1.1.2.3; 1.2.1.3; 1.2.2.3
            print("xd>xb &","vdx < vbx &"," vbx<0 & ","xd-xb>S_max")
            print("here 4a","xb =", xb, "; xd =", xd, "; vbx =", vbx, "; vdx =", vdx)
            xbb = xb; xdd = xd; vvbx = vbx; vvdx = vdx
            cdis1 = np.abs(xd-xb-((vdx**2-vmax**2)/(2*a_min))-((vmax**2-vbx**2)/(2*a_max)))
            str1 = "here 4a3: xd>xb & vdx > vbx & vbx<0 & S_min<(xd-xb)<s_max "
            xb = CA_CV_CA(str1,a_max,vbx,vmax,vmax,cdis1,a_min,vmax,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx) 
    else: #xd<xb
        if np.abs(xd-xb) <= S_min:
            if vbx>=0:
                if vdx>vbx:
    # Section 2.1.1.1
                    cdis = ((xb-xd)+S_min)
                    str1 = "here - 5a: xd<xb & vdx > vbx & vbx>0 & (xd-xb)<S_min "
                    xb  = CA_CV_CA(str1,a_min,vbx,-vbx,-vbx,-cdis,a_max,-vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                else:#(vdx<=vbx)
    # Section 2.2.1.1
                    cdis = (-(xb-xd)+S_min)  
                    str1 = "here 5bb:xd<xb & vdx < vbx & vbx<0 & (xd-xb)<S_min "
                    xb  = CV_CA_CA(str1,cdis,vbx,a_min,vbx,-vbx,(a_max if vdx>-vbx else a_min),-vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)          
            else:#(vbx<0)
                if vdx>=vbx:
    # Section 2.1.2.1
                    cdis = S_min+(xb-xd)
                    str1 = "here 5b:xd>xb & vdx < vbx & vbx>0 & (xd-xb)<S_min vdx<0"
                    xb  = CV_CA_CA(str1,-cdis,vbx,a_max,vbx,-vbx,(a_min if vdx<-vbx else a_max),-vbx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
                else:#(vdx<=vbx)
    # Section 2.2.2.1
                    cdis = S_min+(xb-xd)
                    print("here 5ba","xb =", xb, "xd =", xd, "vbx =", vbx, "vdx =", vdx, "cdis", cdis)
                    str1 = "here 5ba: xd>xb & vdx < vbx & vbx<0 & S_min>(xd-xb) "
                    xb  = CV_CA_CA_CA(str1,-cdis,vbx,a_max,vbx,-vbx,a_max,-vbx,-vdx,
                                a_min,-vdx,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
        elif np.abs((xd-xb)) <= S_max:
            if vbx>=0:
    # Section 2.1.1.2 ; 2.2.1.2
                print("here 6","xb =", xb,"xd =", xd,"vbx =", vbx,"vdx =", vdx,"S_max", S_max, "S_min", S_min)
                print("The distance is reachable. Accn -- deceleration")
                VV1 = ((2 * a_max * a_min * (xd - xb) + a_min * vbx**2 - a_max * vdx**2) / (a_min - a_max))
                VV2 = ((2 * a_max * a_min * (xd - xb) + a_max * vbx**2 - a_min * vdx**2) / (a_max - a_min))
                print("VV1:", VV1, "VV2", VV2)
                if (VV1>=vbx**2):
                    VV = np.sqrt(VV1)
                    a1 = a_min
                else:
                    VV = np.sqrt(VV2)
                    a1 = a_max
                print("VV:", VV, "a1", a1)
                str1 = "here 6:  xd<xb & vdx > vbx & vbx>0 & S_min<(xd-xb)<S_max"
                xb = CA_CA_CA(str1,a_min,vbx,-vbx,-a1,-vbx,-VV,
                                       a1,-VV,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
            else:#(vbx<0)
    # Section 2.1.2.2 and 2.2.2.2
                VV1 = ((2 * a_max * a_min * (xd - xb) + a_min * vbx**2 - a_max * vdx**2) / (a_min - a_max))
                VV2 = ((2 * a_max * a_min * (xd - xb) + a_max * vbx**2 - a_min * vdx**2) / (a_max - a_min))
                if (VV1>=vbx**2):
                    VV = np.sqrt(VV1)
                    a1 = a_max
                else:
                    VV = np.sqrt(VV2)
                    a1 = a_min
                str1 = "here 6b:  xd<xb & vdx > vbx & vbx<0 & S_min<(xd-xb)<S_max"
                xb = CA_CA(str1,a1,vbx,-VV,-a1,-VV,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
        else: #(xd-xb)>S_max
    # Section 2.1.1.3 ; 2.1.2.3 ; 2.2.1.3 and 2.2.2.3
            str1 = "here 6b:  xd<xb & vdx > vbx & vbx<0 & (xd-xb)>S_max"
            cdis1 = np.abs(xd-xb-((vdx**2-vmin**2)/(2*a_max))-((vmin**2-vbx**2)/(2*a_min)))
            str1 = "here 4a3: xd>xb & vdx > vbx & vbx<0 & S_min<(xd-xb)<s_max "
            xb = CA_CV_CA(str1,a_min,vbx,vmin,vmin,-cdis1,a_max,vmin,vdx,xb,xd,vbx,vdx,S_min,S_max, plx, pltt, posx, velx, accx)
####            estop = 1 
    return xb,xd,vbx,vdx,estop

