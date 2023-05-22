import React from 'react'
import { BrowserRouter as Router, Route, Routes} from 'react-router-dom'
import Home from './pages/Home/index'
import Profile from './pages/Profile/index'
import ToExplore from './pages/ToExplore/index'
import ChatBot from './pages/ChatBot/index'
import Notifications from './pages/Notifications/index'
import Bookmarks from './pages/Bookmarks/index'
import Dashboard from './pages/Dashboard/index'
import Settings from './pages/Settings/index'
import Apresentation from './pages/Apresentation/index'
import Signup from './pages/Access/Signup'
import Signin from './pages/Access/Signin'
import './App.css'

// cnd fontweasome
export default () =>{
    return(
        <Router>
            <Routes>
                <Route path='/' element={<Home/>}/>
                <Route path='/profile' element={<Profile/>}/>
                <Route path='/toExplore' element={<ToExplore/>}/>
                <Route path='/chatBot' element={<ChatBot/>}/>
                <Route path='/notifications' element={<Notifications/>}/>
                <Route path='/bookmarks' element={<Bookmarks/>}/>
                <Route path='/dashboard' element={<Dashboard/>}/>
                <Route path='/settings' element={<Settings/>}/>
                <Route path='/apresentation' element={<Apresentation/>}/>
                <Route path='/signin' element={<Signin/>}/>
                <Route path='/signup' element={<Signup/>}/>
            </Routes>
        </Router>
    )
}