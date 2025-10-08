! Routines from Lynn's phaflux code for converting pha channels to physical values

SUBROUTINE CHANNEL_TO_KEV(DET,CHNUM,EKEV)
    implicit none
    INTEGER(KIND=4), INTENT(IN) :: DET
    REAL(KIND=4), INTENT(IN) :: CHNUM
    REAL(KIND=4), INTENT(OUT) :: EKEV

    integer(kind=4) N
    real(kind=4) DENGY, DENGYCHNL, SLOPE
    !   SUBROUTINE TO CONVERT ENERGY CHANNEL NUMBER TO KEV FOR THE THREE AMPTE
    !    DETECTORS.
    !       INPUT:   DET     THE DETECTOR NUMBER, 1,2 OR 3
    !                CHNUM   THE CHANNEL NUMBER TO BE CONVERTED
    !       OUTPUT:  EKEV    THE ENERGY IN KEV
    REAL(kind=4), DIMENSION(4, 28) :: ENGYCHNL
    DATA ENGYCHNL/30.,28.,29.,19., 35.,30.,31.,23., 40.,32.,33.,29., &
            50.,36.,37.,35.5, 60.,40.,41.,41.5, 70.,47.,45.5,47.5, &
            80.,55.,52.5,51., 90.,0.,61.,57., 100.,67.,68.,66., &
            120.,80.,81.,79., 140.,92.,95.,93., 160.,106.,108.,106., &
            180.,119.,121.,119., 200.,132.,134.,132.5, 250.,165.5,167.5, &
            166., 300.,199.,200.5,201.5, 400.,266.,266.,266., 500.,333.5, &
            333.,0., 600.,400.5,399.,399.5, 700.,467.,466.,0., 800.,534., &
            532.5,534., 900.,601.,599.,0., 1000.,668.,666.,668., &
            1100.,735.,732.,0.,1200.,801.,799.,802., 1300.,868.,866.,0., &
            1400.,934.,932.,936., 1500.,1000.5,999.,1003./
    !
    EKEV = 0
    DO N=2,28
        IF (CHNUM > ENGYCHNL(DET+1,N)) CYCLE
        IF (ENGYCHNL(DET+1,N-1) /= 0) THEN
            DENGY = ENGYCHNL(1,N) - ENGYCHNL(1,N-1)
            DENGYCHNL = ENGYCHNL(DET+1,N) - ENGYCHNL(DET+1,N-1)
        ELSE
            DENGY = ENGYCHNL(1,N) - ENGYCHNL(1,N-2)
            DENGYCHNL = ENGYCHNL(DET+1,N)  - ENGYCHNL(DET+1,N-2)
        END IF
        SLOPE = DENGY / DENGYCHNL
        EKEV = SLOPE * (CHNUM - ENGYCHNL(DET+1,N)) + ENGYCHNL(1,N)
        EXIT
    end do
END SUBROUTINE CHANNEL_TO_KEV

!****************************************************************

SUBROUTINE CHANNEL_TO_NS(CHNUM,TNS)
    implicit none
    REAL(KIND=4), INTENT(IN) :: CHNUM
    REAL(KIND=4), INTENT(OUT) :: TNS
    !
    !  MODIFICATION HISTORY
    !     2 AUG 85  CHANGED BACK TO 320 NS MAX, FROM 323 NS MAX
    !
    TNS = CHNUM * 320. / 1023.
END SUBROUTINE CHANNEL_TO_NS

!****************************************************************

SUBROUTINE MASS(E,T,M)
    !  WRITTEN BY LYNN M KISTLER    22-APR-85
    !       LAST UPDATE: 4-NOV-85
    !
    !  INPUT:  E - ENERGY IN KEV
    !          T - TIME IN NS
    !          DPU - 'D' IF DPU ALGORITHM HAS BEEN REQUESTED
    !                'S' IF DPU ALGORITHM IS USED TO CALCULATE MASS ONLY
    !                'R' IF THE REGULAR MASS ALGORITHM IS USED
    !          CNO - 'Y' IF A 'CNO FIT' IS REQUESTED.  A 'CNO FIT' USES
    !                 JUST THE C, N, O, AND Ne CURVES TO FIT THE DATA
    !  OUTPUT: M - MASS IN AMU
    ! jonv - Fortunately phaflux.for only uses the DPU algorithm,
    ! jonv -   because some of this other code calls IMSL routines
    ! jonv -   for which there is no source code!
    ! jonv - Also, the CNO mode is never used from phaflux.for either.
    implicit none
    REAL(KIND=4), INTENT(IN) :: E, T
    REAL(KIND=4), INTENT(OUT) :: M
    REAL(KIND=4) :: DPUA(6)
    REAL(KIND=4) :: LNMASS,LNEMEAS,LNT
    DATA DPUA/-0.0148377,-0.423249,-1.63108,0.306567,0.0379776, 0.0592979/

    LNEMEAS = ALOG(E)
    LNT = ALOG(T)
    LNMASS = DPUA(1) + DPUA(2)*LNEMEAS + DPUA(3)*LNT + &
            DPUA(4)*LNEMEAS*LNT + DPUA(5)*LNEMEAS**2 + &
            DPUA(6)*LNT**3
    M = EXP(LNMASS)
END SUBROUTINE MASS

!****************************************************************
! jonv - this is the Greenspan version of this routine. It has two
! jonv - more parameters than the Kistler version.
! jonv - phaflux.for requires a version with these two extra paramters.
!
SUBROUTINE MASSPERQ(T,M,DV,PAPS,MPQ)!,DPU,RDV)
    implicit none
    !
    !  PROGRAMMER:  LYNN M KISTLER
    !  WRITTEN:     25-APR-85
    !!
    !  SUBROUTINE TO FIND THE MASS PER CHARGE OF A PULSE HEIGHT EVENT FOR
    !  AMPTE/CHEM.
    !  INPUT:    T - TIME OF FLIGHT IN NANOSECONDS
    !            M - MASS, IN AMU, AS CALCULATED FROM THE MASS SUBROUTINE.
    !                IF M = 0, THERE WAS NO MEASURED ENERGY.
    !           DV - THE DEFLECTION VOLTAGE STEP INDEX (0:31) FOR THIS EVENT
    !         PAPS - THE CURRENT PAPS VOLTAGE(KEV)
    !          DPU - 'D' IF DPU ALGORITHM HAS BEEN REQUESTED -> M, MPQ, EPQ
    !              - 'E' DPU ALGORITHM BUT USE REVISED OR MEASURED VOLTAGES
    !                'S' DPU ALGORITHM IS USED TO CALCULATE M, BUT NOT MPQ OR EPQ
    !                'R' THE REGULAR (?) ALGORITHMS FOR M AND M/Q ARE USED
    !                'C' USE APPROXIMATE POWER-LAW ESTIMATE OF STOPPING POWER-SPC
    !                'B' FORCE USE OF REVISED/MEASURED VOLTAGES IN 'D' OR 'C'
    !          RDV - 'Y' IF THE REVISED DEFLECTION VOLTAGES SHOULD BE USED
    !                'N' IF THE 'MEASURED' VOLTAGES SHOULD BE USED
    !  OUTPUT: MPQ - THE MASS PER CHARGE OF THE INCIDENT ION
    !
    REAL(kind=4), INTENT(IN) :: T,M,PAPS
    BYTE, INTENT(IN) :: DV
    REAL(kind=4), INTENT(OUT) :: MPQ
    !
    REAL(kind=4) :: EPQ,EPN(11),MASS,REPQ(0:63)
    REAL(kind=4) :: MEPQ(0:63),S,MPQ1,MPQ2
    REAL(kind=4) :: CFDEPQDX
    EXTERNAL CFDEPQDX
    INTEGER(kind=2) :: CHECK
    integer(kind=4) :: N
    real(kind=4) :: DEDX, DE
    real(kind=4) :: cfoil

    REAL(KIND=4), PARAMETER :: DX = 4.9     ! carbon foil thickness in micrograms/cm**2
    ! phafulx only calls MASSPERQ with DPU=S and RDV=Y
    CHARACTER, PARAMETER :: DPU = "S", RDV = "Y"
    DATA MEPQ/ &
            1.4,1.52,1.65,1.77,1.94,2.12,2.31,2.51,2.72,2.94,3.2,3.5, &
            3.79,4.12,4.5,4.91,5.11,5.58,6.07,6.6,7.2,7.85,8.54,9.3, &
            10.2,11.05,12.0,13.0,14.2,15.5,16.9,18.5,20.9,22.8,24.9, &
            27.2,29.7,32.4,35.2,38.4,41.8,45.6,49.8,54.4,59.4,64.7, &
            70.5,76.8,81.4,88.7,96.7,104.,115.,125.,137.,149.,163., &
            179.,193.,211.,230.,251.,274.,300./
    DATA REPQ/ &
            1.4,1.52,1.65,1.77,1.94,2.12,2.31,2.51,2.72,2.94,3.2,3.5, &
            3.79,4.12,4.5,4.91,5.11,5.58,6.07,6.6,7.2,7.85,8.54,9.3, &
            10.2,11.05,12.0,13.0,14.2,15.5,16.9,18.5,20.9,22.32,24.9, &
            26.47,29.7,32.03,35.2,37.89,41.8,44.46,49.8,53.04,59.4,63.8, &
            70.5,75.89,81.4,88.56,96.7,105.47,115.,126.05,137.,150.92, &
            163.,177.12,193.,210.49,230.,251.26,274.,299./
    !
    !  CHECK IF THE DPU ALGORITHM IS REQUESTED.  IF SO CALCULATE M/Q USING IT.
    !
    IF (DPU.EQ.'D'.OR.DPU.EQ.'d'.OR. &
            DPU.EQ.'E'.OR.DPU.EQ.'e') THEN
        S = DV
        EPQ = 1.400 * (2 **((S-1)/8))
        IF(DPU.EQ.'E'.OR.DPU.EQ.'e') THEN
            IF (RDV.EQ.'Y'.OR.RDV.EQ.'y') THEN
                EPQ = REPQ(DV)
            ELSE
                EPQ = MEPQ(DV)
            END IF
        ENDIF
        MPQ = 1.7401E-5 * (EPQ + PAPS - 2.0) * (T**2)
        !       MPQ = 1.7401E-5 * (EPQ + PAPS - 0.0) * (T**2)
        GO TO 102
    ELSE IF (DPU.EQ.'C'.OR.DPU.EQ.'c') THEN
        S = DV
        EPQ = 1.400 * (2 **((S-1)/8))
        IF (RDV.EQ.'Y'.OR.RDV.EQ.'y') THEN
            EPQ = REPQ(DV)
        ELSE
            EPQ = MEPQ(DV)
        END IF
        MPQ = 1.7401E-5 * (EPQ + PAPS - 2.0) * (T**2)
        IF(MPQ.GT.0..AND.M.GT.0.) THEN
            1002   mpq2 = 1.7401e-5*(epq + paps - dx*CFDEPQDX(m,epq,mpq))*t**2
            !                               if((mpq2-mpq)/sqrt(mpq*mpq2)).gt.0.05) then
            mpq = mpq2
            !                               goto 1002
            !                               endif
        ELSE
            MPQ = 0.
        ENDIF
        GO TO 102
    END IF
    CHECK = 0
    DO N=1,11
        EPN(N) = 0.
    END DO
    IF (RDV.EQ.'Y'.OR.RDV.EQ.'y') THEN
        EPQ = REPQ(DV)
    ELSE
        EPQ = MEPQ(DV)
    END IF
    MASS = M
    !
    !  FOR EVENTS WITH NO ENERGY, GET AN ESTIMATE FOR THE MASS TO BE USED IN
    !  C-FOIL BASED ON THE TIME OF FLIGHT EXPECTED FOR MAJOR SPECIES.
    !
    IF (T.EQ.0.) THEN
        MPQ = 0.
        RETURN
    END IF
    IF (M.LT.0.1) THEN
        MPQ1 = (T/239.89)**2 * (EPQ + PAPS - 4.0)
        IF (MPQ1.LE.1.5) THEN
            MASS = 1.0
        ELSE IF (MPQ1.LE.6.0) THEN
            MASS = 3.97
            !        ELSE IF (MPQ1.LE.23.0) THEN
            !          MASS = 15.87
        ELSE
            !          MASS = 29.8
            MASS = 15.87
        END IF
    END IF
    !
    80    EPN(1) = (239.89/T)**2
    DO N = 1,10
        DEDX = CFOIL(EPN(N),MASS)
        DE = DEDX * DX
        EPN(N+1) = EPN(1) + DE/MASS
        !        WRITE(6,90) N,EPN(N),M,MASS,DV,EPQ
        !90      FORMAT(' N = 'I2,5X,'EPN = 'F7.2,3X,'MASSES: '2F7.2,
        !     1            4X,'DV: ',I6,4X,'E/Q(DV): 'F7.2)
        IF (ABS(EPN(N+1) - EPN(N)).LE.0.01) GO TO 100
    END DO
    N = 10
    100   CONTINUE
    MPQ = (EPQ+PAPS)/EPN(N)
    !
    !   CHECKS IF THE MASS PER CHARGE AND MASS INDICATE THAT THE ION IS IN THE
    !   OXYGEN RANGE.  IF SO, RECALCULATES THE MASS PER CHARGE ASSUMING THAT
    !   THE MASS IS 15.87 (OXYGEN, IN IPAVICH UNITS).
    !
    IF ((CHECK.EQ.0).AND.(M.NE.0)) THEN
        CHECK = 1
        IF (((MPQ.GT.6).AND.(MPQ.LT.23)).OR. &
                !234567
                ((MPQ.LT.6).AND.(M.GT.7))) THEN
            MASS = 15.87
            GO TO 80
        END IF
    END IF
    102   CONTINUE
    MPQ = MAX(0.0,MPQ)
    RETURN
END SUBROUTINE MASSPERQ

real(kind=4) function CFDEPQDX(m,epq,mpq)       ! ESTIMATE ENERGY LOSS IN CARBON FOIL
    implicit none
    real(kind=4), INTENT(IN) :: m,epq,mpq
    CFDEPQDX = 0.1381 * m**0.608 * (epq/mpq)**0.432
    CFDEPQDX = CFDEPQDX * (MPQ/M)
    return
end function CFDEPQDX

!****************************************************************

      REAL(kind=4) FUNCTION CFOIL(EPN,ZAMU)
      implicit none
      REAL(KIND=4), INTENT(IN) :: EPN, ZAMU
      REAL(KIND=4) :: ERG, ZM
      integer*4 I, J
      REAL(KIND=4) :: ZM1, ZM2, XNUM, XDEN, SLOPE, DEDX1, DEDX2, E1, E2, RHS

! INPUT : EPN , incident energy per nuc. in keV per nucleon
!         ZAMU , if > 0, represents  mass in amu
!              , if < 0, then ABS(ZAMU) represents nuclear charge
! OUTPUT : CFOIL = dE/dx  in  keV per microgram/cm**2
      REAL(KIND=4) :: ENERGY(20) , DEDX(20,10)
      REAL(KIND=4) :: RMASS(10) , Z(10) , ZMASS(10)
      INTEGER  ZMFLAG , EFLAG
!
      DATA RMASS/ 1.0 , 3.97 , 11.92 , 13.90 , 15.87 , 20.02 , &
         31.81 , 39.63 , 55.41 , 83.14 /
!
      DATA  Z/ 1., 2., 6., 7., 8., 10., 16., 18., 26., 36. /
!
      DATA ENERGY/1., 1.5, 2.5, 4., 6., 8., 10., 15., 25., 40., 60., &
        80., 100., 150., 250., 400., 600., 800., 1000., 10000. /
!**
!
! PROTONS
      DATA DEDX/.16, .18, .23, .28, .34, .38, .42, .50, .60, .70, .75, &
        .75, .725, .64, .51, .40, .32, .27, .23, .04, &
! HELIUM &
       .43, .48, .57, .66, .75, .84, .91, 1.05, 1.25, 1.47, 1.65, &
        1.82, 1.86, 1.93, 1.81, 1.53, 1.24, 1.04, .89, .16, &
! CARBON &
       1.28, 1.38, 1.52, 1.72, 1.91, 2.10, 2.28, 2.60, 3.25, 4.10, 4.95, &
        5.58, 6.05, 6.75, 7.20, 7.10, 6.80, 6.25, 5.80, 1.45, &
! NITROGEN &
       1.41, 1.51, 1.71, 1.91, 2.18, 2.39, 2.58, 2.97, 3.67, 4.55, 5.50, &
        6.25, 6.80, 7.95, 8.70, 8.80, 8.30, 7.75, 7.20, 1.95, &
! OXYGEN &
       1.42, 1.52, 1.72, 1.99, 2.29, 2.51, 2.73, 3.22, 4.05, 5.00, 6.02, &
        7.00, 7.62, 8.90, 10.1, 10.2, 9.80, 9.35, 8.85, 2.50, &
! NEON &
       1.30, 1.37, 1.54, 1.80, 2.10, 2.35, 2.60, 3.13, 4.15, 5.27, 6.50, &
        7.58, 8.56, 10.7, 13.6, 15.2, 15.2, 14.4, 13.4, 3.90, &
! SULFUR &
       2.00, 2.00, 2.18, 2.53, 3.04, 3.50, 3.91, 4.80, 6.20, 7.70, 9.50, &
        11.0, 12.2, 15.0, 19.0, 22.2, 23.8, 23.8, 23.2, 8.60, &
! ARGON &
       3.08, 3.08, 3.18, 3.45, 3.80, 4.15, 4.45, 5.20, 6.30, 7.77, 9.60, &
        11.3, 12.8, 16.3, 21.0, 24.5, 26.6, 26.8, 26.3, 10.4, &
! IRON &
       3.25, 3.25, 3.25, 3.40, 3.72, 4.13, 4.60, 5.65, 7.40, 9.48, 11.8, &
        13.9, 15.9, 20.9, 27.3, 33.1, 36.9, 38.0, 37.9, 18.3, &
! KRYPTON &
       4.10, 4.10, 4.10, 4.20, 4.43, 4.78, 5.19, 6.30, 8.38, 10.8, 13.6, &
        16.1, 18.5, 24.8, 33.7, 42.4, 48.6, 51.0, 51.5, 29.3 /
      ERG = EPN
      ZM = ZAMU
      ZMFLAG = 0
      EFLAG = 0
      DO 90 I=1,10
              ZMASS(I) = RMASS(I)
              IF(ZM .LT. 0.)   ZMASS(I) = Z(I)
 90   CONTINUE
      ZM = ABS(ZM)
      IF(ZM .LT. 1.)   ZM = 1.
      IF(ERG .LT. 1.E-5)   ERG = 1.E-5
      DO 140 I=1,10
              IF(ZM .EQ. ZMASS(I))   GO TO 145
              IF( I .EQ. 1 )         GO TO 140
              IF(ZM .LT. ZMASS(I))   GO TO 150
 140  CONTINUE
      I = 10
      GO TO 150
 145  ZMFLAG = 1
 150  DO 190 J=1,20
              IF(ERG .EQ. ENERGY(J))   GO TO 195
              IF( J .EQ. 1 )           GO TO 190
              IF(ERG .LT. ENERGY(J))   GO TO 200
 190  CONTINUE
      J = 20
      GO TO 200
 195  EFLAG = 1
 200  IF( ZMFLAG*EFLAG .EQ. 1 )  GO TO 600
      IF(ZMFLAG .EQ. 1)  GO TO 300
      IF(EFLAG .EQ. 1)   J = J + 1
      ZM1 = ZMASS(I-1)
      ZM2 = ZMASS(I)
      XNUM = DEDX(J-1,I) - DEDX(J-1,I-1)
      XDEN = ZM2 - ZM1
      SLOPE = XNUM / XDEN
      DEDX1 = DEDX(J-1,I-1) + SLOPE * (ZM-ZM1)
         IF(EFLAG .EQ. 1)   GO TO 500
      XNUM = DEDX(J,I) - DEDX(J,I-1)
      SLOPE = XNUM / XDEN
      DEDX2 = DEDX(J,I-1) + SLOPE * (ZM-ZM1)
      GO TO 350
 300  DEDX1 = DEDX(J-1,I)
      DEDX2 = DEDX(J,I)
 350  E2 = ENERGY(J)
      E1 = ENERGY(J-1)
      XNUM = ALOG( DEDX2/DEDX1 )
      XDEN = ALOG( E2/E1)
      SLOPE = XNUM / XDEN
      RHS = SLOPE * ALOG( ERG/E1 )
      CFOIL = DEDX1 * EXP( RHS )
      RETURN
 500  CFOIL = DEDX1
      RETURN
 600  CFOIL = DEDX(J,I)
      RETURN
      END FUNCTION CFOIL
