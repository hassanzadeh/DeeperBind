#!/usr/bin/perl -w

use Cwd 'realpath';

use File::Path qw(rmtree);
use warnings;
use strict;
use File::Basename;
use Cwd 'abs_path';

die ("Usage: $0 out_dir filter\n") if (@ARGV < 1);

my $outdir= shift (@ARGV);
$outdir = realpath($outdir);
die if (!-d $outdir);

my $filter= shift (@ARGV);
$filter='' if (!$filter);
my %list;
if (-f $filter){
	open IN, "<$filter" or die ($filter);
	my $line;
	while ($line=<IN>){
		chomp $line;

		if ($line=~/^([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t/){
			$list{"$1*$2*$3*$4*$5*$6*$7"}=1;
		}
	}
	close IN;
}

foreach my $m  ('CNN_LSTM','CNN') {
	my $log_dir = "$outdir/$m/";
	mkdir($log_dir) ; #if (! -d $log_dir);
	$log_dir.="/logs/";
	#die if (-d $log_dir);

	mkdir($log_dir) ; #if (! -d $log_dir);
	mkdir($log_dir."/../Results") ;

	my $job_dir = "$outdir/$m/jobs/";
	#die if (-d $job_dir);
	mkdir($job_dir) ;#if (! -d $job_dir);
}

my $walltime="4:00:00";

my @lrs = (0.01,0.001,0.0001);
my @drs = (0.0000001,0.0001,0);
my @wds =  (0.00001,0);
my @bss = (40,100);
my 	@dos=(0,0.1,0.2);
my @models = ('CNN_LSTM','CNN');
my @pbms=('Oct-1','CEH-22','Zif268','Rap1');
#my @pbms=('Cbf1');
my $ppn=1;

for my $lr (@lrs) {
	for my $dr (@drs){
		for my $wd (@wds){
			for my $bs (@bss){
				for my $pbm (@pbms){
					for my $model (@models) {

						my $job_dir = "$outdir/$model/jobs/";
						my $log_dir = "$outdir/$model/logs/";
						my @lstm_layers;
						if ($model eq 'CNN_LSTM'){
							@lstm_layers = ("30,20","30","20","10,10","10,20");
						}
						else{
							@lstm_layers = ("-");
						}

						for my $do (@dos){
							for my $layers (@lstm_layers){
								if (-f $filter){
									if (!$list{"$pbm*$model*$lr*$dr*$wd*$bs*$layers"}){
										next;
									}
								}
								my $f="";
								if ($model eq 'CNN_LSTM') {
									$f="$pbm\_$model\_lr_$lr\_dr_$dr\_wd_$wd\_do_$do\_bs_$bs\_lstm_$layers";
								}else{
									$f="$pbm\_$model\_lr_$lr\_dr_$dr\_wd_$wd\_do_$do\_bs_$bs";
								}
								open OUT , ">$job_dir/$f.PBS";
								print OUT "# Job id: $f\n".
									"#PBS -N $f\n".
									"#PBS -l nodes=1:ppn=$ppn\n".
									"#PBS -l mem=10gb\n".
									"#PBS -l walltime=$walltime\n".
									"#PBS -q iw-shared-6\n".
									"#PBS -j oe\n".
									"#PBS -p 0\n".
									"#PBS -o $log_dir/$f.log\n".
									"#PBS -m abe\n".
									"module unload python ;\n".
									"module load python/3.4 cmake/3.2.0 openblas/0.2.13 cuda/7.5.18 lapack/3.6.0 torch/7\n".
									"cd /nv/hp16/hhassanzadeh3/data/Projects/DeepBind/src\n".
									"echo \"Started on `/bin/hostname`\"\n".
									"th run.lua --model $model --pbm $pbm -r $lr -o $do -w $wd -d $dr -b $bs -t $ppn --optMethod rmsprop --lstm_layers $layers --output_dir $job_dir/../Results > $log_dir/$f.out\n".
									"echo \"Job Ended\"\n";

								close OUT;	
								while (`qstat -u hhassanzadeh3  | grep ' Q'| wc -l ` + `qstat -u hhassanzadeh3 | grep ' R' | wc -l` > 2500){
							    	print ('Queue size: '. `qstat -u hhassanzadeh3  | grep ' Q'| wc -l`);
								    `sleep 20`;
								}


								`qsub  $job_dir/$f.PBS\n`;
								print "Job:$job_dir/$f.PBS submitted\n";
							}
						}
					}
				}
			}
		}
	}
}

