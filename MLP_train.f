        program MLP_train
        implicit none

        !---------------------------------------------------------------------------
	! MLP_train.f
	! Authors: Nabhonil Kar (nkar@princeton.edu) &
	!	   Francisco J. Tapiador (francisco.tapiador@uclm.es)
        !---------------------------------------------------------------------------

        ! GENERAL VARIABLES
        integer i,j,k,ii,jj,kk,iii,ik
        integer it,dum,x
        integer hg, hgs				! to allocate mem and small mem
        parameter(hg=99999, hgs=1000)
        double precision s, sigmoid
        integer patron(hg)
        character*2 numero

	! TIMING VARIABLES
	double precision start_time, stop_time

        !DATASET STRUCTURE
        integer iterations			! number of iterations
        integer ntrain				! number of training samples
	parameter(ntrain=100)

        ! NET ARCHITECTURE
        integer ni				! number of input neurons
        integer nh				! number of hidden layers; cannot exceed hgs
        integer no				! number of output neurons
        !integer nip				! reserve memory
        parameter(ni=784, nh=1, no=10)
	integer n1, n2, n3			! number of neurons in hidden layers, update as needed
	parameter(n1=30, n2=200, n3=100)	! values cannot exceed hgs

        ! NEURONS
        double precision input(hg,ni)
        double precision hidden(nh,hgs)
        double precision output(no)
        double precision target(hg,no)

        ! WEIGHTS AND SUMS FOR THE HIDDEN LAYER
        double precision  w_h(nh,hgs,hgs)
        double precision dw_h(nh,hgs,hgs)
	double precision  b_h(nh,hgs)
	double precision db_h(nh,hgs)

        ! WEIGHTS AND SUMS FOR THE OUTPUT LAYER
        double precision  w_o(hgs,no)
        double precision dw_o(hgs,no)
	double precision  b_o(no)
	double precision db_o(no)

        ! ERROR DELTAS
        double precision delta_h(nh,hgs), delta_o(no)

        ! TRAINING PARAMS
        double precision rate, sse, lambda
	
	! CHARACTER FOR READING INPUT
	character n

	! PARAMETERS FOR HIDDEN LAYER
	integer nnh(nh)				! array of hidden layer lengths
	nnh(1)=n1				! update as needed
!	nnh(2)=n2
!	nnh(3)=n3
	
        !---------------------------------------------------------------------------
        
	! DEFAULT VALUES
        iterations	= 20			! the total number of iterations
        rate		= 0.35d0		! the learning rate
	lambda 		= 1.0d0			! the regularization parameter
        
	!---------------------------------------------------------------------------

        ! READ THE TRAINING/TESTING DATASET
	call cpu_time(start_time)
	open(1, file='./input/train-images.idx3-ubyte', form='unformatted',
     +		access='direct', status='old', recl=1)
	do i = 0, ntrain-1
		do j = 1, 784
			read(1, rec=16+784*i+j) n
			input(i+1, j) = dble(ichar(n))/2.55d2	! must divide by 225 to normalize
								! to [0,1]
		enddo
	enddo
	close(1)
	
	data target /(hg*no)*0.0d0/		! sets whole target array to 0.0d0
	open(1, file='./input/train-labels.idx1-ubyte', form='unformatted',
     +		access='direct', status='old', recl=1)
	do i = 1, ntrain
		read(1, rec=8+i) n
		target(i, ichar(n)+1) = 1.0d0
	enddo
	close(1)

        !---------------------------------------------------------------------------

        ! INIT WEIGHTS, BIASES AND DELTAS
        do i = 1, nh
        	do j = 1, nnh(i)
			if (i .EQ. 1) then
				do k = 1, ni
					dw_h(i,k,j) = 0.0d0
					w_h(i,k,j) = 2.0d0*1/sqrt(dble(ni))*(rand()-0.5d0)
				enddo			
			else
				do k = 1, nnh(i-1)
       					dw_h(i,k,j) = 0.0d0
        				w_h(i,k,j) = 2.0d0*1/sqrt(dble(nnh(i-1)))*(rand()-0.5d0)
        			enddo
			endif
			db_h(i,j) = 0.0d0
			b_h(i,j) = 0.0d0
        	enddo
	enddo

        do k = 1, no
        	do j = 1, nnh(nh)
        		dw_o(j,k) = 0.0d0
        		w_o(j,k) = 2.0d0*1/sqrt(dble(nnh(nh)))*(rand()-0.5d0)
        	enddo
		db_o(k) = 0.0d0
		b_o(k) = 0.0d0
        enddo

	!---------------------------------------------------------------------------

	! START TRAINING
	write(*,*) "Total iterations: ", iterations

        do it=1,iterations

        ! THE LEARNING SCHEDULE
        if (it.eq.int(iterations/2))rate=rate/2.0
        if (it.eq.int(iterations/4))rate=rate/5.0

        ! RANDOMIZE THE INPUTS
        do i=1,ntrain
        	patron(i)=i
        enddo
        do i=1,ntrain
        	ii= i+rand()*((ntrain-i)*1.0d0)
        	iii= patron(i)
        	patron(i)=patron(ii)
        	patron(ii)=iii
        enddo

        ! ITERATE OVER THE ENTIRE TRAINING DATASET
        sse=0.0d0
        do dum=1, ntrain
	x=patron(dum)

	!---------------------------------------------------------------------------

	! COMPUTE HIDDEN NEURON ACTIVATIONS
	do k = 1, nh
		do i = 1, nnh(k)
			s = b_h(k,i)
			if (k .EQ. 1) then
				do j = 1, ni
					s = s + (input(x,j)*w_h(k,j,i))
				enddo
			else
				do j = 1, nnh(k-1)
					s = s + (hidden(k-1,j)*w_h(k,j,i))
				enddo
			endif
			hidden(k,i) = sigmoid(s)
		enddo
	enddo

	!---------------------------------------------------------------------------

	! COMPUTE OUTPUT NEURON ACTIVATIONS
	do k=1,no
		s=b_o(k)
		do j=1,nnh(nh)
			s = s + (hidden(nh,j)*w_o(j,k))
		enddo
		output(k)=sigmoid(s)
	enddo

	!---------------------------------------------------------------------------

	! CALCULATE THE OUTPUT ERROR, WEIGHT CHANGES & BIAS CHANGES
	do k=1,no
		sse=sse+0.5*(target(x,k)-output(k))*(target(x,k)-output(k))
		delta_o(k)=(output(k)-target(x,k))
		db_o(k) = rate*delta_o(k)
		do j=1,nnh(nh)
			dw_o(j,k)=rate*hidden(nh,j)*delta_o(k)
		enddo
	enddo

	!---------------------------------------------------------------------------

 	! CALCULATE THE HIDDEN LAYER ERROR (VIA BACKPROPOGATION), WEIGHT CHANGES & BIAS CHANGES
	do k = nh, 1, -1
		do i = 1, nnh(k)
	       		! CALCULATE DELTAS
			s = 0.0d0
			if (k .EQ. nh) then
		       		do j = 1, no
		       			s = s + (w_o(i,j)*delta_o(j))
	       			enddo
			else
				do j = 1, nnh(k+1)
					s = s + (w_h(k+1,i,j)*delta_h(k+1,j))
				enddo
			endif
			delta_h(k,i)=s*hidden(k,i)*(1.0d0-hidden(k,i))
			! CALCULATE WEIGHT CHANGES
			db_h(k,i) = rate*delta_h(k,i)
			if (k .EQ. 1) then
				do j = 1, ni
					dw_h(k,j,i) = rate*input(x,j)*delta_h(k,i)
				enddo
			else
				do j = 1, nnh(k-1)
					dw_h(k,j,i) = rate*hidden(k-1,j)*delta_h(k,i)
				enddo
			endif
		enddo
	enddo

	!---------------------------------------------------------------------------

	! UPDATE THE WEIGHTS & BIASES 
	! HIDDEN LAYERS
	do k = 1, nh
		do i = 1, nnh(k)
			b_h(k,i) = b_h(k,i) - rate * db_h(k,i)
			if (k .EQ. 1) then
				do j = 1, ni
					w_h(k,j,i) = (1-rate*lambda/dble(ntrain)) * w_h(k,j,i)
     +						- rate * dw_h(k,j,i)
				enddo
			else
				do j = 1, nnh(k-1)
					w_h(k,j,i) = (1-rate*lambda/dble(ntrain)) * w_h(k,j,i)
     +						- rate * dw_h(k,j,i)
				enddo
			endif
		enddo
	enddo
	! OUTPUT LAYER
	do k=1,no
		b_o(k) = b_o(k) - rate * db_o(k)
		do j=1,nnh(nh)
			w_o(j,k) = (1-rate*lambda/dble(ntrain)) * w_o(j,k) - rate * dw_o(j,k)
		enddo
	enddo
	
	enddo	!end of training over all training samples

        ! CHECK TRAINING	
        if (mod(it,1) .EQ. 0) then
		write(*,*) "Iteration: ", it, " SSE: ", sse, " rate: ",rate
	endif
	
	enddo	!end of training over all epochs

	!---------------------------------------------------------------------------

	! WRITE DOWN THE WEIGHTS & BIASES
	open(1,file='./weights/weights.txt')
	do i=1,nh
		do j=1,nnh(i)
			if (i .EQ. 1) then
				do k=1,ni
					write(1,*)w_h(i,k,j)
				enddo
			else
				do k=1,nnh(i-1)
					write(1,*)w_h(i,k,j)
				enddo
			endif			
		enddo
	enddo
	do i = 1,nh
		do j=1,nnh(i)
			write(1,*)b_h(i,j)
		enddo
	enddo
	do i=1,nnh(nh)
		do j=1,no
			write(1,*)w_o(i,j)
		enddo
	enddo
	do i = 1, no
		write(1,*)b_o(i)
	enddo
	close(1)

	call cpu_time(stop_time)
	write(*,*) "Time elapsed: ", stop_time - start_time

	stop
        end

	!---------------------------------------------------------------------------
	!---------------------------------------------------------------------------

	!---------------------------------------------------------------------------
	!	Returns the sigmoid function's value of input x.
	double precision function sigmoid(x)
	double precision x

	sigmoid = 1.0d0/(1.0d0+exp(-x))
	
	return
	end
	!---------------------------------------------------------------------------
