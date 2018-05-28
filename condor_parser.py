#!/usr/bin/env python3

# Condor Parser for RNS v4 and HTCondor 8.6.1
# Prepare a bunch of condor.sumbit and then submit them

# with a little help of my friend sbozzolo
# https://github.com/Sbozzolo/RNSA/blob/master/condor_parser.py
import argparse
import sys
import os
import shutil
import subprocess
from subprocess import Popen, PIPE
import time
import datetime
import itertools as it
import numpy as np

# Parse command line arguments

parser=argparse.ArgumentParser(description='Opinion dynamics simulator')
parser.add_argument('kind', type=str, help='Flavor of the simulation')
parser.add_argument('-K', '--choices', type=int, nargs='+', help='Number of choices', required='True' )
parser.add_argument('-N', '--agents', type=int, nargs='+', help='Number of agents', required='True')
parser.add_argument('-S', '--entropy', type=float, nargs='+', help='Entropy cutoff (to be substituted by initial overlap)', required='True')
parser.add_argument('-pI', '--mediainfluence', nargs='+', type=float, help='Probability of interaction with external media', default=0.)
parser.add_argument('-t', '--steps', type=int, help='Number of steps of the simulation', default=1000)
parser.add_argument('-ot', '--threshold', nargs='+', type=float, help='Clustering threshold', default=[0.8])
parser.add_argument('-P', '--mediapolarization', nargs='+', type=float, help='Polarization of media information', default=[0.8])
parser.add_argument('-a', '--alpha', nargs='+', type=float, help='Flexibility of the agents', default=[0.01])
parser.add_argument('-n', '--special', nargs='+', type=float, help='Fraction of special agents', default=[0.])
parser.add_argument('-pa', '--pagree', type=float, nargs='+', help='Probability of agreement for fixedp agents', default=0.5)
parser.add_argument('-f', '--file', help='where to write the output')
parser.add_argument('-m', '--multiple', action='store_true', help='Use multiple information sources')
parser.add_argument('-g', '--graph', help='where to print the graph')
parser.add_argument("-o", "--out", type = str,
                    help = "set output folder")
parser.add_argument("-c", "--condor", action='store_true', help = "launch the jobs aftear creating the submit script")

    
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

kind=args.kind                 # flavor of the simulation
K=args.choices                # number of choices                    [2,+inf)
N=args.agents                 # number of agents                     [0,+inf)
S=args.entropy                # entropy cutoff                       [0,lnK]
p_I=args.mediainfluence       # influence of external media          [0,1]
t=args.steps                  # interaction time                     [0,+inf)
o_t=args.threshold            # clustering threshold                 [0,1] 
P=args.mediapolarization      # polarization of external information [0,1]
a=args.alpha                  # flexibility of the agents            [0,1]
W=args.special
pa=args.pagree
f=args.file                   
g=args.graph
o=args.out
m=args.multiple

# End parsing CLI arguments

pwd=subprocess.run('pwd',stdout=subprocess.PIPE)
pwd=pwd.stdout.decode('utf-8')
pwd=pwd.strip('\n')

#exec_path=pwd+'/special2.py'
exec_path=pwd+'/job.sh'
#exec_path = "/home/claudiochi/Documents/TesiMagistrale/special.py"

# # Define a new folder based on time if -o is not provided
## --- Not presently required
# # basefolder is where data will be stored, full path

# if (args.out != None):
#     basefolder = args.out
# else:
#     now = datetime.datetime.now()
#     basefolder = "{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute)

# # Add absolute path
# basefolder = os.path.join(os.getcwd(), basefolder)
#basefolder = "/home/claudio/repo_tesi/results"
#basefolder = "/home/claudiochi/Documents/TesiMagistrale/results/"

basefolder=pwd+"/results/"
#In farm e` /home/claudiochi/TesiMagistrale

# Remove old results, if they exist
#if os.path.exists(basefolder):
#    shutil.rmtree(basefolder)
#os.mkdir(basefolder)


### MY ATTEMPT
#print(S)
#print(N)
#for x,y,z in it.product(np.arange(K[0],K[1],1),
#                        np.arange(N[0],N[len(N)],np.abs((N[0]-N[len(N)])/5)),
#                        np.arange(S[0],S[1],0.1)):

#Obtain hostname
host=subprocess.run('hostname',stdout=subprocess.PIPE)
host=host.stdout.decode('utf-8')
host=host.strip('\n')


for k,n,s,pi,ot,p,al,w,pag in it.product(K,N,S,p_I,o_t,P,a,W,pa):
    fileext="K"+str(k)+".N"+str(n)+".S"+str(s)+".pI"+str(pi)+".ot"+str(ot)+".P"+str(p)+".a"+str(al)+".n"+str(w)+".pa"+str(pag)
    arguments=kind
    if m:
        arguments=arguments+" -m "
        fileext=fileext+".m"
    graph="graph."+fileext+".svg"
    file="output."+fileext+".dat"
    arguments = arguments \
                + " -K "   + str(k)  \
                + " -N "  + str(n)  \
                + " -S "  + str(s)  \
                + " -pI " + str(pi) \
                + " -ot " + str(ot) \
                + " -P "  + str(p)  \
                + " -t "  + str(t)  \
                + " -a "  + str(al) \
                + " -n "  + str(w)  \
                + " -pa " + str(pag)  \
                + " -f "  + str(file)  \
                + " -g "  + str(graph)  
    
    #----Write to file
    parentfolder=basefolder
    condorfolder=os.path.join(parentfolder, "condor/")
    if not os.path.exists(condorfolder):
        os.mkdir(condorfolder)
    condorfilename = os.path.join(condorfolder, "condor."+fileext+".submit")
    condorfile = open(condorfilename, "w")
    # Write Condor file
    print ("Executable   = " + exec_path, file = condorfile)
    print ("Universe     = Vanilla", file = condorfile)
    print ("InitialDir   = " + parentfolder, file = condorfile)
    
#    print("Requirements = TARGET.UidDomain == \"heisenberg.pcteor1.mi.infn.it\" && TARGET.FileSystemDomain == \"heisenberg.pcteor1.mi.infn.it\" ", file = condorfile)

    print("should_transfer_files = YES", file = condorfile)
    print("when_to_transfer_output = ON_EXIT", file = condorfile)
    print("transfer_input_files = "+pwd+"/python3.tar.gz, "+pwd+"/echocABM/functions.py, "+pwd+"/echocABM/model.py, "+pwd+"/main.py", file = condorfile)

#    print ("Requirements = (Machine != \""+host+"\")", file = condorfile)
    # If a job is running for more than 10 minutes, kill it
    #print ("periodic_remove = JobStatus == 2 && CurrentTime-EnteredCurrentStatus > 300", file = condorfile)
    # print ("Notification = Complete", file = condorfile)
    # print ("notify_user  = spammozzola@gmail.com", file = condorfile)
    print ("", file = condorfile)
    print ("Output   = " + parentfolder+"main.out."+fileext, file = condorfile)
    print ("Error   = " + parentfolder+"main.err."+fileext, file = condorfile)
    print ("Log   = " + parentfolder+"main.log."+fileext, file = condorfile)
    print ("", file = condorfile)
        
    print ("Arguments   = " + arguments, file = condorfile)
    print ("Queue", file = condorfile)
    #print("Output   = main."+ out.+" ", file = condorfile)
    condorfile.close()

    #Execute
    if args.condor:
        # This is not the best wat to do that, but it works
        result = subprocess.check_output("condor_submit " + condorfilename, shell = True)
        # Delete condor.submit
        #os.remove(condorfilename)
        # Print to STDIO results
        print(result.decode('utf8'))
    else:
        print ("condor.submit have been produced")
        
#sys.exit(0)
