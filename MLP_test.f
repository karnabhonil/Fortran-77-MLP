        program MLP_test
        implicit none

	!----------------------------------------------------------------------
	! MLP_test.f
	! Author: Nabhonil Kar(nkar@princeton.edu) &
	!	  Francisco J. Tapiador (francisco.tapiador@uclm.es)
	!----------------------------------------------------------------------


	! GENERAL VARIABLES
	integer i,j,k,ii,jj,kk,iii,ik
	integer it,dum,x
	integer hg, hgs				! to allocate mem and small mem
	parameter(hg=99999, hgs=2000)
	double precision s, sigmoid
	double precision sse

	! DATASET STRUCTURE
	integer ntest				! number of testing samples
	parameter(ntest=5000)

	! NET ARCHITECTURE
	integer ni    				! number of input neurons
	integer nh				! number of hidden layers; cannot exceed hgs
	integer no				! number of output neurons
	parameter(ni=784, nh=1, no=10)
	integer n1, n2, n3			! number of neurons in hidden layers, update as needed
	parameter(n1=100, n2=100, n3=100)	! values cannot exceed hgs

	! NEURONS
	double precision test(hg,ni)
	double precision output(hg,no)
	double precision output_net(hg,no)
	double precision hidden(nh,hgs)

	! WEIGHTS & SUMS FOR THE HIDDEN LAYER
	double precision w_h(nh,hgs,hgs)
	double precision b_h(nh,hgs)

	! WEIGHTS & SUMS FOR THE OUTPUT LAYER
        double precision w_o(hgs,no)
	double precision b_o(no)

	! CHARACTER FOR READING INPUT
	character n
	! PARAMETERS FOR HIDDEN LAYER
	integer nnh(nh)				! array of hidden layer lengths
	nnh(1)=n1				! update as needed
!	nnh(2)=n2
!	nnh(3)=n3

	!----------------------------------------------------------------------

	! READ THE TESTING DATASET
	open(1, file='./input/t10k-images.idx3-ubyte', form='unformatted',
     +		access='direct', status='old', recl=1)
	do i = 0, ntest-1
		do j = 1, 784
			read(1, rec=16+784*i+j) n
			test(i+1, j) = dble(ichar(n))
		enddo
	enddo
	close(1)
	
	data output /(hg*no)*0.0d0/		! sets whole output array to 0.0d0
	open(1, file='./input/t10k-labels.idx1-ubyte', form='unformatted',
     +		access='direct', status='old', recl=1)
	do i = 1, ntest
		read(1, rec=8+i) n
		output(i, ichar(n)+1) = 1.0d0
	enddo
	close(1)

        write(*,*)"Testing cases: ",ntest

	!----------------------------------------------------------------------
       
	! READ THE WEIGHTS
	open(1,file='./weights/weights.txt')
	do i = 1, nh
		do j = 1, nnh(i)
			if (i .EQ. 1) then
				do k = 1, ni
					read(1,*) w_h(i,k,j)
				enddo
			else
				do k = 1, nnh(i-1)
					read(1,*) w_h(i,k,j)
				enddo
			endif
		enddo
	enddo
	do i = 1, nh
		do j = 1, nnh(i)
			read(1,*) b_h(i,j)
		enddo
	enddo
	do i = 1, nnh(nh)
		do j = 1, no
			read(1,*) w_o(i,j)
		enddo
	enddo
	do i = 1 , no
		read(1,*) b_o(i)
	enddo
        close(1)

	!----------------------------------------------------------------------

	! TESTING PHASE
	sse = 0.0d0
	do x = 1, ntest
		do k = 1, nh		
			do i = 1, nnh(k)
				s = b_h(k,i)
				if (k .EQ. 1) then
					do j = 1, ni
						s = s+(test(x,j)*w_h(k,j,i))
					enddo
				else
					do  j = 1, nnh(k-1)
						s = s+(hidden(k-1,j)*w_h(k,j,i))
					enddo
				endif
				hidden(k,i) = sigmoid(s)
			enddo
		enddo
		do k = 1, no
			s = b_o(k)
			do j = 1, nnh(nh)
				s = s+hidden(nh,j)*w_o(j,k)
			enddo
			output_net(x,k) = sigmoid(s)
			sse = sse+0.5*(output_net(x,k)-output(x,k))*(output_net(x,k)-output(x,k))
		enddo
	enddo

	!----------------------------------------------------------------------
        
	! WRITE DOWN TEST RESULTS
        open(1,file='./output/recon_net_norm.txt')
        open(2,file='./output/recon_ori_norm.txt')
        do x = 1, ntest
        	write(1,*) (output_net(x,k),k=1,no)	! net guess
        	write(2,*) (output(x,k),k=1,no)		! original data
        enddo
        close(1)
        close(2)

	!----------------------------------------------------------------------

	! PRINT FRACTION OF CORRECTLY IDENTIFIED TEST SAMPLES
	! (net guess is defined as output neuron with highest activation)
	ii = 0
	do i = 1, ntest
		call maxi(output, hg, no, i, j)
		call maxi(output_net, hg, no, i, k)
		if (j .EQ. k) then
			ii = ii + 1
		endif
	enddo
	write (*,*), "Percent correctly identified: ", dble(ii)/dble(ntest)*100, "%"
	write (*,*), "SSE ", sse

	stop
	end

	!----------------------------------------------------------------------
	!----------------------------------------------------------------------

	!----------------------------------------------------------------------
	!	Returns the sigmoid function's value of input x.
	double precision function sigmoid(x)
	double precision x

	sigmoid = 1.0d0/(1.0d0+exp(-x))

	return
	end
	!----------------------------------------------------------------------
	
	!----------------------------------------------------------------------
	!	Traverses row i of A(m,n) to find the max value.
	!	Returns the index where the max value is found through ret.
	subroutine maxi(A, m, n, i, ret)
	integer m, n, i, ret
	double precision A(m,n)
	
	integer j
	double precision max
	ret = 0
	max = 0.0d0
	
	do j = 1, n
		if (A(i,j) .GT. max) then
			ret = j
			max = A(i,j)
		endif
	enddo
	
	return
	end
	!----------------------------------------------------------------------
