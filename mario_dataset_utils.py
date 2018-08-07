import os
import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
import cv2
import scipy.misc
import scipy.signal
import csv
import random
import math
from scipy import stats

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

dicEvents={'311':'ARMORED_TURTLE_KILLSTOMP','411':'JUMP_FLOWER_KILLSTOMP','511':'CANNON_BALL_KILLSTOMP','611':'CHOMP_FLOWER_KILLSTOMP'
    ,'321':'ARMORED_TURTLE_UNLEASHED','421':'JUMP_FLOWER_UNLEASHED'
    ,'521':'CANNON_BALL_UNLEASHED','621':'CHOMP_FLOWER_UNLEASHED','120':'LARGE_Mode_START','220':'FIRE_Mode_START'
    ,'-19':'win','09':'killedbyturtle','19':'killedbygoompa','29':'killedbygap','39':'killedbyflower','49':'killedbycannon'}

player_names=['_dit_22_02_1_A','_dit_22_02_1_B','_teo_10_03_1_A','ama_16_03_1_A',
      'ama_16_03_1_B','ant_23_03_1_B','asi_08_02_1_A','asi_08_02_1_B','ben_23_03_1_A','ben_23_03_1_B',
      'chr_22_03_1_A','chr_22_03_1_B','cla_22_03_1_A','cla_22_03_1_B','dan_17_11_1_A','dan_17_11_1_B','dar_22_03_1_A',
      'dar_22_03_1_B','des_16_03_1_A','des_16_03_1_B','des_22_02_1_A','des_22_02_1_B','ele_07_12_1_A','ele_07_12_1_B',
      'ele_10_03_1_A','ele_10_03_1_B','ema_14_02_1_A','ema_14_02_1_B','gog_10_03_1_A','gog_10_03_1_B','gou_10_03_1_A',
      'gou_10_03_1_B','hec_22_03_1_A','hec_22_03_1_B','ili_07_12_1_A','ili_07_12_1_B','jon_23_03_1_A','jon_23_03_1_B',
      'jua_22_03_1_A','jua_22_03_1_B','jul_22_02_1_A','jul_22_02_1_B','jul_23_03_1_A','jul_23_03_1_B','kal_08_02_1_A',
      'kal_08_02_1_B','kal_16_03_1_A','kal_16_03_1_B','kar_14_03_1_A','kar_14_03_1_B','kat_16_03_1_A','kat_16_03_1_B',
      'kun_22_03_1_A','kun_22_03_1_B','leo_22_03_1_A','leo_22_03_1_B','mar_1_2_1_A','mar_1_2_1_B','mar_22_02_1_A',
      'mar_22_02_1_B','mar_23_03_1_A','mar_23_03_1_B','mrn_22_02_1_A','mrn_22_02_1_B','nat_22_02_1_A','nat_22_02_1_B',
      'nel_1_2_1_A','nel_1_2_1_B','nti_16_03_1_A','nti_16_03_1_B','par_16_03_1_A','par_16_03_1_B','pen_17_11_1_A',
      'pen_17_11_1_B','pin_1_2_1_A','pin_1_2_1_B','ril_22_02_1_A','ril_22_02_1_B','sar_10_03_1_A','sar_10_03_1_B',
      'sig_22_02_1_A','sig_22_02_1_B','sim_14_03_1_A','sim_14_03_1_B','ska_16_03_1_A','ska_16_03_1_B','sof_08_02_1_A',
      'sof_08_02_1_B','sta_10_03_1_A','sta_10_03_1_B','tas_16_03_1_A','tas_16_03_1_B',
      'the_08_02_1_A','the_08_02_1_B',
      '_dit_22_02_2_A','_dit_22_02_2_B','ama_16_03_2_A',
      'ama_16_03_2_B','ant_23_03_2_A','ant_23_03_2_B','asi_08_02_2_A','asi_08_02_2_B','ben_23_03_2_B',
      'chr_22_03_2_A','chr_22_03_2_B','cla_22_03_2_A','cla_22_03_2_B','dan_17_11_2_B','dar_22_03_2_A',
      'dar_22_03_2_B','des_16_03_2_A','des_16_03_2_B','des_22_02_2_A','des_22_02_2_B','ele_07_12_2_A','ele_07_12_2_B',
      'ele_10_03_2_A','ele_10_03_2_B','ema_14_02_2_A','ema_14_02_2_B','gog_10_03_2_A','gog_10_03_2_B',
      'gou_10_03_2_B','ili_07_12_2_A','ili_07_12_2_B','jon_23_03_2_A','jon_23_03_2_B',
      'jua_22_03_2_A','jua_22_03_2_B','jul_22_02_2_A','jul_22_02_2_B','jul_23_03_2_A','jul_23_03_2_B','kal_08_02_2_A',
      'kal_08_02_2_B','kal_16_03_2_A','kal_16_03_2_B','kar_14_03_2_A','kar_14_03_2_B','kat_16_03_2_A','kat_16_03_2_B',
      'kun_22_03_2_A','kun_22_03_2_B','leo_22_03_2_A','leo_22_03_2_B','mar_1_2_2_A','mar_1_2_2_B','mar_22_02_2_A',
      'mar_22_02_2_B','mar_23_03_2_A','mar_23_03_2_B','mrn_22_02_2_A','mrn_22_02_2_B','nat_22_02_2_A','nat_22_02_2_B',
      'nel_1_2_2_A','nti_16_03_2_A','nti_16_03_2_B','par_16_03_2_A','par_16_03_2_B',
      'pen_17_11_2_B','pin_1_2_2_A','pin_1_2_2_B','ril_22_02_2_A','ril_22_02_2_B','sar_10_03_2_A','sar_10_03_2_B',
      'sig_22_02_2_A','sig_22_02_2_B','sim_14_03_2_A','sim_14_03_2_B','ska_16_03_2_A','ska_16_03_2_B','sof_08_02_2_A',
      'sof_08_02_2_B','sta_10_03_2_A','sta_10_03_2_B','tas_16_03_2_A','tas_16_03_2_B',
      'the_08_02_2_A','the_08_02_2_B']

path_images='video_frames/'

def get_number_of_total_frames(path):
    total_num_images=0
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        total_num_images=total_num_images+1
    return total_num_images 
   
def get_frame_number(path,time):
    file = open(path+'/length.txt', 'r') 
    total_length_ms=float(file.readline())
    return int((get_number_of_total_frames(path)/total_length_ms)*time)

def predict_expressions(output_number,path_predictions,player_name,event,number_of_previous_frames,number_of_next_frames,normalize):
    valids=[False]*(number_of_next_frames+number_of_previous_frames)
    predictions=np.ones((number_of_next_frames+number_of_previous_frames,output_number))
    predictions=predictions*-10
    count=0
    count2=0
    if event-number_of_previous_frames+1<=0:
        for i in range(np.abs(event-number_of_previous_frames+1)):
            count2=count2+1
    if normalize:
        save_address=path_predictions+'/normalized_signals/'+player_name+'.csv'
    else:
        save_address=path_predictions+'/filtered_signals/'+player_name+'.csv'
    for line in open(save_address):
        if count in range(np.max((event-number_of_previous_frames+1,1)),event+number_of_next_frames+1):
            prediction=[float(l) for l in line.split(',')]
            if -10 not in prediction:
                valids[count2]=True
            predictions[count2,:]=np.array(prediction)
            count2=count2+1
        count=count+1
    valids=np.array(valids)
    return predictions,valids

def get_events_dict():
    events_dicts=[]
    player_ids=[]
    for name in player_names:
        for filename in glob.glob(os.path.join(path_images+name, '*.csv')):
            events_dict={}
    #         for key in dicEvents:
    #             events_dict[dicEvents[key]]=[]
            events_dict['win']=[]
            events_dict['killedby']=[]
            events_dict['killedbygap']=[]
            events_dict['killedbycannon']=[]
            events_dict['killedbygoompa']=[]
            events_dict['killedbyflower']=[]
            events_dict['killedbyturtle']=[]
            events_dict['killed']=[]
            events_dict['changemode']=[]
            save=False
            for line in open(filename):    
                row=line.split(',')
                if save:
                    save=False
                    if 'killedby' in dicEvents[saved_event]:
                        events_dict[dicEvents[saved_event]].append(get_frame_number(path_images+name,float(row[0])))
                        events_dict['killedby'].append(get_frame_number(path_images+name,float(row[0])))
                    elif 'KILLSTOMP' in dicEvents[saved_event]:
                        events_dict['killed'].append(get_frame_number(path_images+name,float(row[0])))
                    elif 'win' in dicEvents[saved_event]:
                        events_dict['win'].append(get_frame_number(path_images+name,float(row[0])))
                    elif 'Mode' in dicEvents[saved_event]:
                        events_dict['changemode'].append(get_frame_number(path_images+float(row[0])))
                k=row[1][0:len(row[1])-1].strip()
                if k in dicEvents:
                    if row[0]!='':
                        if 'killedby' in dicEvents[k]:
                            events_dict[dicEvents[k]].append(get_frame_number(path_images+name,float(row[0])))
                            events_dict['killedby'].append(get_frame_number(path_images+name,float(row[0])))
                        elif 'KILLSTOMP' in dicEvents[k]:
                            events_dict['killed'].append(get_frame_number(path_images+name,float(row[0])))
                        elif 'win' in dicEvents[k]:
                            events_dict['win'].append(get_frame_number(path_images+name,float(row[0])))
                        elif 'Mode' in dicEvents[k]:
                            events_dict['changemode'].append(get_frame_number(path_images+name,float(row[0])))
                    else:
                        save=True
                        saved_event=k
            events_dicts.append(events_dict)
    return events_dicts  

def get_affect_before_after_event(pre,post,T,reaction_time,player_index,emotion,output_number,path_predictions,normalize):
    response_,valid_=predict_expressions(output_number,path_predictions,player_names[player_index],T+reaction_time,pre,post,normalize) 
    half=pre
    return get_average_of_valids(response_[0:half,emotion]),get_average_of_valids(response_[half:,emotion])
    
def get_average_of_valids(y):
    count_t=0
    sum_t=0
    #number of time window
    for i in range(y.shape[0]):
        if y[i]!=-10:
            sum_t=sum_t+y[i]
            count_t=count_t+1
    if count_t!=0:
        return float(sum_t)/float(count_t)
    else:
        return -10

def get_valid_signal(signal):
    valid_signal=[]
    for k in range(signal.shape[0]):
        if signal[k]!=-10:
            valid_signal.append(signal[k])
    return valid_signal

def get_affect_around_event(pre,post,T,reaction_time,player_index,emotion,ax,l,output_number,path_predictions,normalize):
    response_,valid_=predict_expressions(output_number,path_predictions,player_names[player_index],T+reaction_time,pre,post,normalize)
    valid_signal=get_valid_signal(response_[:,emotion])
    
    signal=response_[:,emotion]
    if -10 not in signal:
        avg_of_signal=np.average(signal)
        signal_mean_subtracted=signal-avg_of_signal
        variance_of_signal=get_std_of_valids(signal,avg_of_signal)
        signal_scaled_by_variance=signal_mean_subtracted/variance_of_signal
        line,=ax.plot(np.linspace(-15,30,45),signal_mean_subtracted,label=str(T))
        """slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(signal_mean_subtracted)),signal_mean_subtracted)
        line_ = slope*range(len(signal_mean_subtracted))+intercept
        print('slope: ',slope)
        ax.plot(np.linspace(-15,30,45),line_)"""
        l.append(line)
    ax.set_xlabel('frames')
    ax.set_ylabel(get_expressions_dict(output_number)[emotion])
    """if output_number>2:
        ax.set_yticks(np.linspace(0.0, 1.0, num=10))
    else:
        ax.set_yticks(np.linspace(-1.0, 1.0, num=10))"""
    ax.set_xticks(np.linspace(-15,30,10))
    return get_average_of_valids(response_[:,emotion]),np.amin(signal_mean_subtracted),np.amax(signal_mean_subtracted)

def get_affect_per_player_per_event(path_save_plot,player_index,event_name,output_number,pre,post,path_predictions,normalize):
    events_dicts=get_events_dict()
    total_effects=[]
    avg_affect=[]
    min_y=1000
    max_y=-1000
    if events_dicts[player_index][event_name]!=[]:
        if output_number>2:
            #fig,axs = plt.subplots(3,3,figsize=(15,15))
            fig,ax = plt.subplots(figsize=(5,5))
        else:
            fig,axs = plt.subplots(1,2,figsize=(8,5))
        for i in [3]:#range(output_number):
            sum_affect=0
            count_affect=0
            Ts=[]
            l=[]
            for index,T in enumerate(events_dicts[player_index][event_name]):
                if True:
                #if T!=-10:
                    Ts.append(str(T))
                    if output_number>2:
                        affect,min_,max_=get_affect_around_event(pre,post,T,0,player_index,i,ax,l,output_number,
                                                             path_predictions,normalize)
                    else:
                        affect=get_affect_around_event(pre,post,T,0,player_index,i,axs[i%2],l,output_number,
                                                             path_predictions,normalize)
                    if min_y>min_:
                        min_y=min_
                    if max_y<max_:
                        max_y=max_
                    if affect!=-10:
                        total_effects.append(affect)
                        sum_affect=sum_affect+affect
                        count_affect=count_affect+1
            if len(l)!=0:
                if output_number>2:
                    #ax.set_yticks(np.linspace(min_y, max_y, num=10))
                    ax.set_yticks(np.linspace(-0.2398, 0.4225, num=10))
                    ax.legend(l,Ts)
                else:
                    ax.set_yticks(np.linspace(-1.0, 1.0, num=10))
                    axs[i%2].legend(l,Ts)

            if count_affect!=0:
                avg_affect.append(sum_affect/float(count_affect))
            else:
                avg_affect.append(None)
        plt.savefig('results/'+path_save_plot+'/'+player_names[player_index]
                    +'_'+event_name+'.png',bbox_inches = 'tight')
        #plt.show()
        plt.close(fig)
        return avg_affect,total_effects
    else:
        return [None]*output_number,total_effects

def save_affect_per_player_per_event(path_save,path_save_plot,output_number,pre,post,path_predictions,normalize):
    for player_index,events_dict in enumerate(get_events_dict()):
        if True:
            event_average_dicts=[]
            for key in events_dict:
                if True:
                    d={}
                    affect,total_effects=get_affect_per_player_per_event(path_save_plot,player_index,key,output_number,pre,post,
                                                                         path_predictions,normalize)
                    half=int(len(total_effects)/output_number)
                    affects=[]
                    for i in range(output_number):
                        affects.append(total_effects[i*half:(i+1)*half])
                    for i in range(output_number):
                        if i==0:
                            affects_=np.array(affects[i])
                        else:
                            affects_=np.column_stack((affects_,np.array(affects[i])))                      
                    for h in range(half):
                        d={}
                        for i in range(output_number):
                            d[get_expressions_dict(output_number)[i]]=affects_[h,i]
                        d['Event']=key+str(h)
                        event_average_dicts.append(d)
            with open('compare_results/mario_dataset/'+path_save+'/'+player_names[player_index]+'.csv', 'w') as csvfile:
                fieldnames =get_fieldsname(output_number)
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for event_average_dict in event_average_dicts:
                    writer.writerow(event_average_dict) 
                
def get_fieldsname(output_number):
    if output_number==2:
        return ['Event','Valence', 'Arousal']
    elif output_number==7:
        return ['Event','Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    elif output_number==8:
        return ['Event','Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral','Contempt']
    
def get_expressions_dict(output_number):
    if output_number==2:
        return {0:'Valence', 1:'Arousal'}
    elif output_number==7:
        return {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    elif output_number==8:
        return {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral',7:'Contempt'}
    
def predict_expressions_for_the_whole_signal(path):
    count_line=0
    predictions=[]
    for line in open(path):
        if count_line!=0:
            prediction=[float(l) for l in line.split(',')]
            predictions.append(np.array(prediction))
        count_line=count_line+1
    return np.array(predictions)

def filter_signal(path_predictions,output_number,kernel_size):
    for name in player_names:
        print(name)
        predictions=predict_expressions_for_the_whole_signal(path_predictions+'/inpainted_signals/'+name+'.csv')
        print('signals: ',predictions.shape)
        signals=[]
        for emotion in range(output_number):
            y=predictions[:,emotion]
            valid_signal=get_valid_signal(y)
            signal=np.zeros((y.shape[0]))
            signal_count=0
            filtered_signal=scipy.signal.medfilt(valid_signal,kernel_size=kernel_size)
            for k in range(y.shape[0]):
                if y[k]!=-10:
                    signal[k]=filtered_signal[signal_count]
                    signal_count=signal_count+1
                else:
                    signal[k]=y[k]
            signals.append(np.array(signal))
        signals=np.array(signals)
        print('filtered_signal: ',signals.shape)
        with open(path_predictions+'/filtered_signals/'+name+'.csv', 'w') as csvfile:
            fieldnames=get_fieldsname(output_number)
            fieldnames.remove('Event')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            count=0
            for i in range(signals.shape[1]):
                row={}
                for i,f in enumerate(fieldnames):
                    row[f]=signals[i,count]
                writer.writerow(row)
                count=count+1   
                
def inpainting_signal(path_predictions,output_number):
    for name in player_names:
        print(name)
        predictions=predict_expressions_for_the_whole_signal(path_predictions+'/'+name+'.csv')
        for emotion in range(output_number):
            prediction=predictions[:,emotion]
            for i,p in enumerate(prediction):
                if p==-10:
                    prediction[i]=np.nan


            ipn_kernel = np.array([1,1,1,1,1]) # kernel for inpaint_nans

            nans = np.isnan(prediction)
            while np.sum(nans)>0:
                prediction[nans] = 0
                vNeighbors = scipy.signal.convolve((nans==False),ipn_kernel,mode='same')
                im2 = scipy.signal.convolve(prediction,ipn_kernel,mode='same')
                im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
                im2[vNeighbors==0] = np.nan
                im2[(nans==False)] = prediction[(nans==False)]
                prediction = im2
                nans = np.isnan(prediction)
            predictions[:,emotion]=prediction
        print('normalized_expressions: ',predictions.shape)        
        with open(path_predictions+'/inpainted_signals/'+name+'.csv', 'w') as csvfile:
            fieldnames=get_fieldsname(output_number)
            fieldnames.remove('Event')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            count=0
            for i in range(predictions.shape[0]):
                row={}
                for i,f in enumerate(fieldnames):
                    row[f]=predictions[count,i]
                writer.writerow(row)
                count=count+1
        
def get_min_signal(y):
    min_=1000
    min_index=-1
    for i in range(y.shape[0]):
        if y[i]!=-10:
            if min_>y[i]:
                min_=y[i]
                min_index=i
    return min_,min_index

def get_max_signal(y):
    max_=-1000
    max_index=-1
    for i in range(y.shape[0]):
        if y[i]!=-10:
            if max_<y[i]:
                max_=y[i]
                max_index=i 
    return max_,max_index 

def get_emotions(cluster_number,total_clusters,t,pre,post,output_number,average_type,path_predictions,
                 list_effect_dicts,list_effect_dicts2,normalize):
    #for all players
    for player_index,events_dict in enumerate(get_events_dict()):
        #for all events
        for key in events_dict.keys():
            for T in events_dict[key]:            
                if T!=-10:
                    signals=[]
                    slopes=[]
                    for emotion in range(output_number):
                        if average_type=='average_around_event':
                            response_,valid_=predict_expressions(output_number,path_predictions,
                                                                 player_names[player_index],T,pre,post,normalize)
                            
                            signal=response_[:,emotion]
                            if -10 not in signal:
                                avg_of_signal=np.average(signal)
                                signal_mean_subtracted=signal-avg_of_signal
                                """variance_of_signal=get_std_of_valids(signal,avg_of_signal)
                                signal_scaled_by_variance=signal_mean_subtracted/variance_of_signal"""
                                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(signal_mean_subtracted)),
                                                                                               signal_mean_subtracted)
                                signals.append(signal_mean_subtracted)
                                slopes.append(slope)
                    for emotion in range(output_number):
                        if signals!=[]:
                            if key not in list_effect_dicts[emotion].keys():
                                list_effect_dicts[emotion][key]=[]
                            list_effect_dicts[emotion][key].append(signals[emotion])     
                        if average_type=='average_around_event':
                            if slopes!=[]:
                                if key not in list_effect_dicts2[emotion].keys():
                                    list_effect_dicts2[emotion][key]=[]
                                list_effect_dicts2[emotion][key].append(slopes[emotion])                  
    return list_effect_dicts,list_effect_dicts2

def plot_(cluster_number,average_type,key,output_number,list_effect_dicts,path_save_plot):
    if output_number>2:
        fig,axs = plt.subplots(2,3,figsize=(18,10))        
    else:
        fig,axs = plt.subplots(1,2,figsize=(8,5))
    min_y=1000
    max_y=-1000
    for emotion in range(output_number-1):
        if True:
            lines=[]
            labels=[]
            for t in range(1):
                key_=key
                if key_ in list_effect_dicts[emotion].keys():
                    affects=np.array(list_effect_dicts[emotion][key_])
                    #median=median_of_multiple_signals(affects)  
                    avg=average_of_multiple_signals(affects)      
                    #median=avg
                    std=std_of_multiple_signals(affects,avg)
                    std=np.array(std)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(avg)),avg)
                    line = slope*range(len(avg))+intercept
                    #print(key,emotion,round(slope,6))
                    if output_number>2:
                        if key_=='killed':
                            l,=axs[int(emotion/3),emotion%3].plot(np.linspace(-15, 30, num=45),avg,label='killing enemy')
                        else:
                            l,=axs[int(emotion/3),emotion%3].plot(np.linspace(-15, 30, num=45),avg,label=key_)
                        axs[int(emotion/3),emotion%3].plot(np.linspace(-15, 30, num=45), line)
                        axs[int(emotion/3),emotion%3].errorbar(np.linspace(-15, 30, num=45),avg,std,ecolor='g')
                        lines.append(l)
                        axs[int(emotion/3),emotion%3].set_xlabel('frames')
                        axs[int(emotion/3),emotion%3].set_ylabel(get_expressions_dict(output_number)[emotion])
                        axs[int(emotion/3),emotion%3].set_xticks(np.linspace(-15, 30, num=10))
                        """axs[int(emotion/3),emotion%3].set_yticks(np.linspace(min(np.amin(avg),np.amin(avg-std)),
                                                                             max(np.amax(avg),np.amax(avg+std)),
                                                                             num=10))"""
                        #axs[int(emotion/3),emotion%3].set_yticks(np.linspace(np.amin(avg),np.amax(avg),num=10))
                        """if min_y>np.amin(avg):
                            min_y=np.amin(avg)
                        if max_y<np.amax(avg):
                            max_y=np.amax(avg)"""
                        
                        if min_y>min(np.amin(avg),np.amin(avg-std)):
                            min_y=min(np.amin(avg),np.amin(avg-std))
                        if max_y<max(np.amax(avg),np.amax(avg+std)):
                            max_y=max(np.amax(avg),np.amax(avg+std))
                    else:
                        l,=axs[emotion%2].plot(np.linspace(-15, 30, num=45),avg,label=key_)
                        axs[emotion%2].plot(np.linspace(-15, 30, num=45), line)
                        axs[emotion%2].errorbar(np.linspace(-15, 30, num=45),avg,std,ecolor='g')
                        lines.append(l)
                        axs[emotion%2].set_xlabel('frames')
                        axs[emotion%2].set_ylabel(get_expressions_dict(output_number)[emotion])
                        axs[emotion%2].set_xticks(np.linspace(-15, 30, num=10))
                        if min_y>min(np.amin(avg),np.amin(avg-std)):
                            min_y=min(np.amin(avg),np.amin(avg-std))
                        if max_y<max(np.amax(avg),np.amax(avg+std)):
                            max_y=max(np.amax(avg),np.amax(avg+std))
                    if key_=='killed':
                        labels.append('killing enemy')
                    else:
                        labels.append(key_)
        if output_number>2:
            axs[int(emotion/3),emotion%3].legend(lines,labels)
        else:
            axs[emotion%2].legend(lines,labels)
    for emotion in range(output_number-1):
        if output_number>2:
            #axs[int(emotion/3),emotion%3].set_yticks(np.linspace(min_y,max_y,num=10))
            axs[int(emotion/3),emotion%3].set_yticks(np.linspace(-0.1353,0.1657,num=10))
        else:
            axs[emotion%2].set_yticks(np.linspace(min_y,max_y,num=10))
        #axs[int(emotion/3),emotion%3].set_yticks(np.linspace(-0.1353,0.1657,num=10))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.5)
    if lines!=[]:
        plt.savefig('results/'+
                    path_save_plot+'/'+average_type+'_'+str(cluster_number)+'_'+key+'.png',bbox_inches = 'tight')
        #plt.show()
        plt.close(fig)
    else:
        plt.close(fig)
        
def plot_3(e,cluster_number,average_type,output_number,list_effect_dicts,list_effect_dicts2,path_save_plot):
    eventslist=['killedby','win','changemode','killed']
    #eventslist=['killedby.0','killedby.1','killedby.2']
    if output_number>2:
        fig,axs = plt.subplots(2,4,figsize=(24,10))        
    else:
        fig,axs = plt.subplots(1,2,figsize=(8,5))
    
    lines=[]
    labels=[]
    min_y=1000
    max_y=-1000
    for key_index,key in enumerate(eventslist):
        for emotion in [e]:#range(output_number):
            if True:
                lines2=[]
                labels2=[]
                for t in range(1):
                    key_=key
                    if key_ in list_effect_dicts[emotion].keys():
                        affects=np.array(list_effect_dicts[emotion][key_])                
                        avg=average_of_multiple_signals(affects) 
                        std=std_of_multiple_signals(affects,avg)
                        std=np.array(std)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(avg)),avg)
                        line = slope*range(len(avg))+intercept
                        if output_number>2:
                            l,=axs[0,key_index].plot(np.linspace(-15, 60, num=75),avg,label=key_)
                            l2,=axs[0,key_index].plot(np.linspace(-15, 60, num=75), line,label=str(round(slope,4)))
                            axs[0,key_index].errorbar(np.linspace(-15, 60, num=75),avg,std,ecolor='g')
                            lines.append(l)
                            lines2.append(l2)
                            labels2.append(str(round(slope,4)))
                            axs[0,key_index].set_xlabel('frames')
                            if key_index==0:
                                axs[0,key_index].set_ylabel(get_expressions_dict(output_number)[emotion])
                            axs[0,key_index].set_xticks(np.linspace(-15, 60, num=6))
                            if key=='killed':
                                axs[0,key_index].set_title('killing enemy')
                            else:
                                axs[0,key_index].set_title(key)
                            axs[0,key_index].legend(lines2,labels2)
                            
                            if min_y>min(np.amin(avg),np.amin(avg-std)):
                                min_y=min(np.amin(avg),np.amin(avg-std))
                            if max_y<max(np.amax(avg),np.amax(avg+std)):
                                max_y=max(np.amax(avg),np.amax(avg+std))
                        else:
                            l,=axs[emotion%2].plot(range(len(affects)),affects,label=key_)
                            lines.append(l)
                            axs[emotion%2].set_xlabel('frames')
                            axs[emotion%2].set_ylabel(get_expressions_dict(output_number)[emotion])
                            axs[emotion%2].set_yticks(np.linspace(-1.0, 1.0, num=10))
    for key_index,key in enumerate(eventslist):
        axs[0,key_index].set_yticks(np.linspace(min_y,max_y,num=10))
        #axs[0,key_index].set_yticks(np.linspace(-0.1863,0.1810,num=10))
    
    min_x=1000
    max_x=-1000
    min_interval=1000
    for key_index,key in enumerate(eventslist):#get_events_dict()[0].keys(): 
        for emotion in [e]:#range(output_number-1):
            if key in list_effect_dicts2[emotion].keys():
                slopes=np.array(list_effect_dicts2[emotion][key])
                if min_x>np.amin(slopes):
                    min_x=np.amin(slopes)
                if max_x<np.amax(slopes):
                    max_x=np.amax(slopes)
                bin_interval=(np.amax(slopes))/5.0 
                if min_interval>bin_interval:
                    min_interval=bin_interval
    for key_index,key in enumerate(eventslist):#get_events_dict()[0].keys(): 
        for emotion in [e]:#range(output_number-1):
            if key in list_effect_dicts2[emotion].keys():
                slopes=np.array(list_effect_dicts2[emotion][key])
                if output_number>2:
                    bins=[]
                    initial_value=0.0
                    bins.append(initial_value)
                    #bin_interval=(np.max(slopes))/5.0
                    bin_interval=min_interval#(max_x)/5.0
                    #for i in range(6):
                    #print(np.amin(slope),np.amax(slope))
                    while initial_value<max_x:
                        initial_value=initial_value+bin_interval
                        bins.append(initial_value)
                    initial_value=0.0
                    #for i in range(6):
                    while initial_value>min_x:
                        initial_value=initial_value-bin_interval
                        bins.append(initial_value)
                    bins.sort()

                    hist_, bin_edges = np.histogram(slopes,bins=bins)
                    bin_edges=np.round(bin_edges,4)
                    labels=[]
                    """for i in range(len(bin_edges)-1):
                        labels.append(str(bin_edges[i])+': '+str(bin_edges[i+1]))"""
                    
                    """num_in_bin=[0]*len(bins)
                    for b in range(len(bins)-1):
                        for s in slopes:
                            if s>=bins[b] and s<bins[b+1]:
                                num_in_bin[b]=num_in_bin[b]+1"""

                    width = 1.0
                    l= axs[1,key_index].bar(np.arange(len(hist_)), hist_/len(slopes), width,color='b')
                    #l= axs[1,key_index].bar(np.arange(len(bins)), np.array(num_in_bin)/len(slopes), width,color='b')
                    axs[1,key_index].set_xticks(np.arange(len(labels)))
                    axs[1,key_index].set_yticks(np.linspace(0.0, 0.6, num=5))
                    axs[1,key_index].set_xticklabels(labels,rotation='vertical')
                    lines.append(l)
                    axs[1,key_index].set_xlabel('g')
                    if key_index==0:
                        axs[1,key_index].set_ylabel('p(g)')
                    #axs[1,key_index].set_title(key)
                    axs[1,key_index].axvline(x=22.4, color='k', linestyle='--',label='0')
                    axs[1,key_index].text(22.2,-0.03,'0')
                else:
                    l,=axs[emotion%2].plot(xx,yy,label=key_)
                    lines.append(l)
                    axs[emotion%2].set_xlabel('frames')
                    axs[emotion%2].set_ylabel(get_expressions_dict(output_number)[emotion])
                    axs[emotion%2].set_yticks(np.linspace(-1.0, 1.0, num=10))
                labels.append(key)
    fig.subplots_adjust(hspace=0.2)
    fig.subplots_adjust(wspace=0.3)
    plt.savefig('results/'+
                path_save_plot+'/histogram_'+average_type+'_'+str(cluster_number)+'__'+key+'.png',bbox_inches = 'tight')
    #plt.show()
    plt.close(fig)
        
def save_plots_3(e,cluster_number,total_clusters,pre,post,output_number,average_type,path_save_plot,path_predictions,normalize):
    list_effect_dicts=[]
    for emotion in range(output_number):
        list_effect_dicts.append({})
    list_effect_dicts2=[]
    for emotion in range(output_number):
        list_effect_dicts2.append({})
    list_effect_dicts,list_effect_dicts2=get_emotions(cluster_number,total_clusters,0,pre,post,output_number,
                                     average_type,path_predictions,list_effect_dicts,list_effect_dicts2,normalize)
    plot_3(e,cluster_number,average_type,output_number,list_effect_dicts,list_effect_dicts2,path_save_plot)
    for key in get_events_dict()[0].keys():
        plot_(cluster_number,average_type,key,output_number,list_effect_dicts,path_save_plot)

def average_of_multiple_signals(signals):
    avg=[]
    #number of time window
    for i in range(signals.shape[1]):
        count_t=0
        sum_t=0
        #number of events
        for j in range(signals.shape[0]):
            if True:#signals[j,i]!=-10:
                sum_t=sum_t+signals[j,i]
                count_t=count_t+1
        if count_t!=0:
            avg.append(float(sum_t)/float(count_t))
        else:
            avg.append(-10)
    return avg

def median_of_multiple_signals(signals):
    median=[]
    #number of time window
    for i in range(signals.shape[1]):
        m=[]
        #number of events
        for j in range(signals.shape[0]):
            if True:#signals[j,i]!=-10:
                m.append(signals[j,i])
            
        if m!=[]:
            median.append(np.median(m))
        else:
            median.append(-10)
    return np.array(median)

def std_of_multiple_signals(signals,avg):
    std=[]
    #number of time window
    for i in range(signals.shape[1]):
        count_t=0
        sum_t=0
        #number of events
        for j in range(signals.shape[0]):
            if True:#signals[j,i]!=-10:
                sum_t=sum_t+(signals[j,i]-avg[i])**2
                count_t=count_t+1
        if count_t!=0:
            std.append(np.sqrt(float(sum_t)/float(count_t)))
        else:
            std.append(-10)
    return std

def get_std_of_valids(y,avg):
    count_t=0
    sum_t=0
    #number of time window
    for i in range(y.shape[0]):
        if y[i]!=-10:
            sum_t=sum_t+(y[i]-avg)**2
            count_t=count_t+1
    if count_t!=0:
        return math.sqrt(float(sum_t)/float(count_t))
    else:
        return -10