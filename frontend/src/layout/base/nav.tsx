import IconBid from '@/assets/layout/bid.svg'
import IconHistory from '@/assets/layout/history.svg'
import IconKnowledge from '@/assets/layout/knowledge.svg'
import IconMemory from '@/assets/layout/memory.svg'
import IconDatabase from '@/assets/layout/database.svg'
import IconNewChat from '@/assets/layout/newchat.svg'
import IconNews from '@/assets/layout/news.svg'
import React, { useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { NavItem } from './nav-item'
import { SessionDrawer } from '@/components/session-drawer'
import './nav.scss'

export function Nav() {
  const { pathname } = useLocation()
  const [sessionDrawerOpen, setSessionDrawerOpen] = useState(false)

  const items = useMemo(
    () => [
      {
        key: 'newchat',
        label: '新的聊天',
        icon: IconNewChat,
        href: '/chat',
      },
      {
        key: 'history',
        label: '对话历史',
        icon: IconHistory,
        href: '#',
        onClick: () => setSessionDrawerOpen(true),
      },
      {
        key: 'memory',
        label: '记忆库',
        icon: IconMemory,
        href: '/memory',
      },
      {
        key: 'knowledge',
        label: '知识库',
        icon: IconKnowledge,
        href: '/knowledge',
      },
      {
        key: 'database',
        label: '数据库',
        icon: IconDatabase,
        href: '/database',
      },
      {
        key: 'news',
        label: '行业资讯',
        icon: IconNews,
        href: '/news',
      },
      {
        key: 'bid',
        label: '招投标信息',
        icon: IconBid,
        href: '/bidding',
      },
      // 暂时隐藏职业规划
      // {
      //   key: 'career',
      //   label: '职业规划',
      //   icon: IconCareer,
      //   href: '#',
      // },
    ],
    [],
  )

  return (
    <>
      <div className="base-layout-nav">
        {items.map(({ key, onClick, ...item }) => (
          <NavItem
            key={key}
            {...item}
            active={pathname === item.href}
            onClick={onClick}
          />
        ))}
      </div>
      <SessionDrawer
        open={sessionDrawerOpen}
        onClose={() => setSessionDrawerOpen(false)}
      />
    </>
  )
}
