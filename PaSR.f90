module extFGH

  interface ! interface for C++ subroutines
  
    subroutine myfgh ( need, nx, x, nf, nh, iusr, rusr, f, g, h ) bind ( c )
      use iso_c_binding
      integer (c_int) :: need(*),nx,nf,nh,iusr(*)
      real (c_double) :: x(*),rusr(*),f(*),g(*),h(*)       
    end subroutine myfgh
	
	subroutine mymix ( nx, ptcl1, ptcl2, alpha, iusr, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx,iusr(*)
      real (c_double) :: ptcl1(*),ptcl2(*),rusr(*),alpha(*)      
    end subroutine mymix
	
	subroutine toxhat ( ptcl, x, nx, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),rusr(*),ptcl(*)      
    end subroutine toxhat
	
	subroutine myfnn ( nx, x, fnn ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),fnn(*)      
    end subroutine myfnn
	
	subroutine fromxhat ( x, ptcl, nx, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),rusr(*),ptcl(*)      
    end subroutine fromxhat
	
  end interface      

end module extFGH

program main

  use extFGH

  implicit none

  integer, pointer :: need(:),need2(:),iusr(:) ! ISAT inputs
  integer :: ii,jj,kk 
  integer :: nTest = 250 ! when storage retrieval accuracy is tested, errors are computed every nTest time steps
  
  integer, parameter :: mode = 2
  ! mode determines the behavior of the code
  ! mode == 1 does no storage retrieval, and outputs a low number (1,000,000) of PaSR samples
  ! mode == 2 performs standard preconditioned ISAT
  ! mode == 3 performs a standalone ISAT simulation
  ! mode == 4 performs a standalone MLP simulation
  
  double precision, parameter :: errtol = 32e-5 ! ISAT error tolerance
  
  integer :: nx=11,nf=11,nh=1 ! dimensions of x and f (h is for an ISAT functionality which is not used here)
  integer :: ns=10, info(100), nPtcl = 10000, idd, nSteps = 4000, counter, i1, i2
  ! ns is the number of species, nPtcl is the number of particles in the reactor, nSteps is the number of time steps
  double precision, pointer    :: x(:), ptcl1(:), ptcl2(:), rusr(:), f(:), fisat(:), fisatsum(:), fnn(:), g(:,:), h(:), ptcls(:,:), alpha(:)
  double precision :: rinfo(70), stats(100), dt, start, finish
  
  double precision :: flowThroughTime, mixTime ! characteristic times which determine the rates at which particles are replaced and mixed
  
  integer :: nH2 = 2, nO2 = 5, nN2 = 11 ! locations of the H2, O2 and N2 mass fractions in the ptcl(:,:) array. ptcl(1,:) is temperature
  ! and the species are arranged in the same order as they are in the chemical mechanism

  double precision :: mFrac = 0.02, pFrac = 0.1, Tfuel = 700.0, Tox = 700.0,error(11)
  ! mFrac controls the mass fraction of the H2 stream in the PaSR, pFrac controls the mass fraction of 
  ! equilibrium combustion products in the PaSR, Tfuel and Tox are the inflow temperatures of the fuel and
  ! oxidizer streams
  
  double precision :: rr, rr2 ! work variables used for random outcomes in particle replacement and mixing
  
  if ( mode.eq.1) then
	nPtcl = 250
	nSteps = 4000
  end if ! for pre-processing we can use significantly less particles

  allocate(need(3),need2(3),iusr(3),x(nx),ptcl1(nx),ptcl2(nx),rusr(2*nx+5),f(nx),fisat(nx),fisatsum(nx),fnn(nx),g(nx,nx),h(1),ptcls(nx+1,nPtcl),alpha(1))
  
  iusr(1) = mode ! used to pass the code's behavior to the C++ routines (determines whether f^{MLP} is subtracted from f(x))
  
  dt = 5e-7 ! single reactor time step
  flowThroughTime = 5e-5 ! characteristic time for particle replacement
  mixTime = 2.5e-5 ! characteristic time for particle mixing

  need = (/ 1, 0, 0 /) ! if we only need f(x) from ISAT
  need2 = (/ 1, 1, 0 /) ! if we need both f(x) and its Jacobian
  
  
  rusr(1:2*nx) = 0.0 ! no normalization initially
  rusr(1:nx+1) = (/ 700.0, 1e-4, 3e-6, 1e-4, 5e-2, 1e-4, 1e-2, 2e-5, 1e-6, 1e-30, 0.753777, 358.9534 /)
  rusr(2*nx+1) = 1e-8 ! absolute error tolerance for the Cantera ODE integration
  rusr(2*nx+2) = 1e-8 ! relative error tolerance for the Cantera ODE integration
  rusr(2*nx+3) = dt ! dt
  rusr(2*nx+4) = 1e-6 ! dx
  rusr(2*nx+5) = 101325.0 ! reactor pressure, in Pa
  
  info  = 0
  rinfo = 0
  info(2)  = 0	! if_g (no piecewise constant approximation of the Jacobian, see ISAT documentation)
  info(12) = 2	! isat op (output ISAT performance data)
  info(28) = 0   ! idites (accuracy test every idites-th ISAT retrieval, not used here)
  rinfo(1) = errtol ! absolute error tolerance which ISAT aims for
  rinfo(2) = errtol ! relative error tolerance which ISAT aims for
  rinfo(3) = 1e2 ! tolerance for EOA growth (not essential here), see ISAT documentation
  rinfo(8) = 3000.  !  stomby - size of ISAT table in MB
  
  error = 0.0 ! initialize the error 
  
  call random_seed(put=(/10,11/)) ! set a random seed for the PaSR reactor
  ! the random seed for ISAT is separate and not set here
  
  call cpu_time(start) ! start the timer
  
  counter = 0
  
  do ii = 1,nPtcl ! begin initialization of particles in the reactor
    ! the particles can come from either the fuel stream, oxidizer stream or equilibrium products stream
  
  
	call random_number(rr)
	call random_number(rr2) 
	
	if ( rr2 < pFrac ) then ! this occurs with probability pFrac
	
		ptcls(:,ii) = (/ 1950.02, 2.01543e-6, 5.04459e-08, 1.88483e-05, 0.114629, 0.000540692, 0.126148, 4.5638e-06, 5.15421e-07, 0.0, 0.753777 /) 
		! phi = 0.5 products at equilibrium	
	
	else
	
		if ( rr < mFrac ) then ! this occurs with probability mFrac
			ptcls(1,ii) = Tfuel
			ptcls(2:nx,ii) = 0.0
			ptcls(nH2,ii) = 1.0		! fuel inflow
		else
			ptcls(1,ii) = Tox
			ptcls(2:nx,ii) = 0.0
			ptcls(nO2,ii) = 1.0/4.76
			ptcls(nN2,ii) = 3.76/4.76	! oxidizer inflow
		end if  
	
	end if
	
	counter = counter + 1
	ptcls(nx+1,ii) = counter	
	
  end do ! end initialization
  
  call toxhat( ptcls(1:nx,1), x, nx, rusr )
  call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h ) ! normalize one particle and call f(x) on it, so that 
  ! the chemistry is initialized
  
  do jj = 1,nSteps ! begin time step
  
	do ii = 1,nPtcl ! perform chemical kinetics on all particles
		call toxhat( ptcls(1:nx,ii), x, nx, rusr ) ! normalize the particle
		
		if ( mode.eq.1 ) then
			print *, x
				call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h )
			print *, f
		end if ! output x and f(x) for the particle, if the code is collecting data for
		! f^{MLP} training
		
		if ( (mode.eq.2).or.(mode.eq.3) ) then
			call isatab( idd,0,nx,x,nf,nh,nh, myfgh, iusr,rusr, info, rinfo, fisat ,g,h,stats)
		end if ! call ISAT, if needed
			
		if ( (mode.eq.2).or.(mode.eq.4) ) then
			call myfnn( nx, x, fnn )
		end if ! call f^{MLP}, if needed
		
		if ( mode.gt.1 ) then ! if the code does storage retrieval
			if (modulo(jj,nTest).eq.0) then 
			
				call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h )
				! every 250-th time step, perform a DE on all
				! particles and calculate the storage retrieval error
			
				do kk = 1,nx
					if ( mode.lt.4 ) then
						error(kk) = error(kk) + (f(kk)-fisat(kk))**2 ! ISAT error,
						! when ISAT is used for storage retrieval
					elseif ( mode.eq.4 ) then
						error(kk) = error(kk) + (f(kk)-fnn(kk))**2 ! f^{MLP} error
						! when f^{MLP} is used for storage retrieval
					end if
						
				end do
			
			end if
		end if
		
		if ( mode.eq.1 ) then ! add the appropriate time increment to the particle's composition
			x = x + f ! DE increment
		elseif ( mode.eq.2 ) then
			x = x + fisat + fnn ! preconditioned ISAT increment
		elseif ( mode.eq.3 ) then 
			x = x + fisat ! standalone ISAT increment
		elseif ( mode.eq.4 ) then
			x = x + fnn ! MLP increment
		end if
		
		call fromxhat( x, ptcls(1:nx,ii), nx, rusr )		
		! convert the particles back to dimensional form
		
	end do ! end of reaction loop
	
	if ((modulo(jj,10).eq.0) .and. (mode.gt.1)) then
		print *, ptcls
	end if
	
	do ii = 1,nPtcl ! start the inflow/outflow loop
	
		call random_number(rr)
		if ( rr < dt/flowThroughTime ) then ! a particle is replaced with probability dt/flowThroughTime
		
			call random_number(rr)		
			call random_number(rr2)
		
				if ( rr2 < pFrac ) then ! same probabilities and compositions of the three streams
					! as when the particles are initialized
	
					ptcls(:,ii) = (/ 1950.02, 2.01543e-6, 5.04459e-08, 1.88483e-05, 0.114629, 0.000540692, 0.126148, 4.5638e-06, 5.15421e-07, 0.0, 0.753777 /) 
					! phi = 0.5 products at equilibrium	
	
				else
	
					if ( rr < mFrac ) then
						ptcls(1,ii) = Tfuel
						ptcls(2:nx,ii) = 0.0
						ptcls(nH2,ii) = 1.0		! fuel inflow
					else
						ptcls(1,ii) = Tox
						ptcls(2:nx,ii) = 0.0
						ptcls(nO2,ii) = 1.0/4.76
						ptcls(nN2,ii) = 3.76/4.76	! oxidizer inflow
					end if  
				
				end if
			
			counter = counter + 1
			ptcls(nx+1,ii) = counter	
			
		end if
	end do
	
	do ii = 1,nPtcl ! begin particle mixing
		call random_number(rr)
		i1 = int(ceiling(rr*real(nPtcl)))
		if (i1 .eq. 0) i1 = 1
		call random_number(rr)
		i2 = int(ceiling(rr*real(nPtcl)))
		if (i2 .eq. 0) i2 = 1 ! pick two particles at random
		
		call random_number(rr)
		
		ptcl1 = ptcls(1:nx,i1)
		ptcl2 = ptcls(1:nx,i2)
		
		call mymix( nx, ptcl1, ptcl2, rr*dt/mixTime, iusr, rusr )
		
		ptcls(1:nx,i1) = ptcl1
		ptcls(1:nx,i2) = ptcl2
		
	end do
  
  end do ! end time step
  
  call cpu_time(finish)
 
  if ( mode.gt.1 ) then
  
	  print *, '("Time = ",e10.3," seconds.")',finish-start
	  
	  print *, 'RMS error: ', sqrt(sum(error)/(nPtcl*nSteps/nTest))
	  
	  print *, 'Mean temp: ', sum(ptcls(1,:)/nPtcl)
	  
	  print *, 'Mean H: ', sum(ptcls(5,:)/nPtcl)
	  
	  if ( (mode.eq.2).or.(mode.eq.3) ) then
	  
		  call isatab( idd,6,nx,x,nf,nh,nh, myfgh, iusr,rusr, info, rinfo, fisat ,g,h,stats)
		  
		  print *, 'Number of queries: ', stats(1)
		  
		  print *, 'Number of leaves: ', stats(12)
		  
		  print *, 'Number of grows: ', stats(4)
		  
		  print *, 'Number of adds: ', stats(5)
		  
		  print *, 'Fraction of retrieves: ', real(stats(2) + stats(3))/real(stats(1))
		  
		  print *, 'Number of unresolved: ', stats(8)
		  
		  print *, 'Number of DEs: ', stats(7)
		  
	  end if
	  
  end if

end program main
