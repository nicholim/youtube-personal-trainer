import os
import subprocess

import urllib.request
from pathlib import Path
import ntpath
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, session
from DTWdistance import distance_using_dtw
from extract_from_user_video import output_user_keypoints
from youtube_links import links
from convert import convert_to_h264
# from werkzeug.utils import secure_filename
# from werkzeug.urls import url_unquote


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ''
# UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__,template_folder='templates')

app.static_folder = ''
# app.static_folder = 'static/uploads/'

app.config['SECRET_KEY'] = "COMP4971C"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('web.html')

@app.route('/', methods=['POST'])
def upload():
    if request.form.get('upload') == "Submit":     
        if 'file' not in request.files:
            flash('No such file')
            return redirect(request.url)
        f = request.files['file']

        if f.filename =='':
            flash('No file selected for uploading')
            return redirect(request.url)
        else:
            # f.filename=f.filename.replace(" ", "%20")
            # f.filename=f.filename.replace("|:", "_")
            # f.filename=f.filename.replace("?&", "")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(file_path)

            return render_template('web.html', filename=f.filename, model=[{'name':'body'}, {'name':'coco'}, {'name':'mpii'}])

    elif request.form.get('analyze') == "Start Analyze":
        model=request.form.get('model_select')
        filename = request.form.get('user_vid')
        user_video_path = filename
        user_out_path = filename[:-4]+"_opose.mp4"
        if(str(model)=="body"):
            user_npy_path_body = filename[:len(filename)-4]+"_body.npy"
            output_user_keypoints(user_video_path, user_npy_path_body, user_out_path, 0.2, model=str(model))
            out_h264 = convert_to_h264(user_out_path)
            rank_result = distance_using_dtw(str(model), user_npy_path_body)
            rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1])[:10])
        elif(str(model)=="coco"):
            user_npy_path_coco = filename[:len(filename)-4]+"_coco.npy"
            output_user_keypoints(user_video_path, user_npy_path_coco, user_out_path, 0.2, model=str(model))
            out_h264 = convert_to_h264(user_out_path)
            rank_result = distance_using_dtw(str(model), user_npy_path_coco)
            rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1])[:10])
        else:
            user_npy_path_mpi = filename[:len(filename)-4]+"_mpii.npy"
            output_user_keypoints(user_video_path, user_npy_path_mpi, user_out_path, 0.2, model=str(model))
            out_h264 = convert_to_h264(user_out_path)
            rank_result = distance_using_dtw(str(model), user_npy_path_mpi)
            rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1])[:10])
        return render_template('web.html', filename=filename, vidanalyzed=out_h264, model=[{'name':'body'}, {'name':'coco'}, {'name':'mpii'}], analyzed=rank_result, links=links)
    else:
        return flash('No file selected for uploading')

@app.route('/display/<filename>')
def display_video(filename):
    # filename=url_unquote(filename)
    # return render_template('web.html', filename=filename, model=[{'name':'body'}, {'name':'coco'}, {'name':'mpi'}])
    return redirect(url_for('static', filename=filename), code=301)

@app.route('/analyzed/<path:vidanalyzed>')
def display_video_analyzed(vidanalyzed):
    # filename=url_unquote(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], vidanalyzed, as_attachment=True)    
    # return redirect(url_for('static', filename=vidanalyzed), code=301)    

##########################

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=7000, debug=False)
